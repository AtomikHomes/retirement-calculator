import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Retirement Planning Calculator",
    page_icon="📈",
    layout="wide",
)


# =========================
# Helpers
# =========================
def fmt_currency(x: float) -> str:
    return f"${x:,.0f}"


def currency_tick(x, pos):
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"


def safe_percent(value: float) -> str:
    return f"{value:.1%}"


# =========================
# Core deterministic model
# =========================
def run_deterministic_projection(
    annual_returns: list[float],
    monthly_income: float,
    monthly_expenses: float,
    non_retirement_savings: float,
    retirement_savings: float,
    current_age: int,
    retirement_growth_end_age: float,
    retirement_access_age: int,
    expected_age_of_death: int,
    expense_reduction_age: int,
    expense_reduction_factor: float,
):
    results = []

    for annual_return in annual_returns:
        liquid_assets = non_retirement_savings
        retirement_assets = retirement_savings
        annual_income = monthly_income * 12
        annual_expenses = monthly_expenses * 12

        ages = []
        liquid_series = []
        total_series = []
        tracker = {}

        hit_zero_year = None
        min_total = liquid_assets + retirement_assets
        min_year = current_age

        for age in range(current_age, expected_age_of_death + 1):
            total_portfolio = liquid_assets + retirement_assets
            tracker[age] = total_portfolio

            ages.append(age)
            liquid_series.append(liquid_assets)
            total_series.append(total_portfolio)

            if total_portfolio < min_total:
                min_total = total_portfolio
                min_year = age

            if total_portfolio < 0 and hit_zero_year is None:
                hit_zero_year = age

            if age == expected_age_of_death:
                break

            # Retirement assets compound until access / end age
            if age < retirement_growth_end_age and retirement_assets > 0:
                retirement_assets *= (1 + annual_return)

            # Move retirement assets into liquid assets at access age
            if age == retirement_access_age and retirement_assets > 0:
                liquid_assets += retirement_assets
                retirement_assets = 0

            # Expense reduction later in life
            if age == expense_reduction_age:
                annual_expenses *= expense_reduction_factor

            # Net cash flow comes out of liquid assets
            net_cash_flow = annual_income - annual_expenses
            liquid_assets += net_cash_flow

            # Liquid assets also compound
            liquid_assets *= (1 + annual_return)

        results.append(
            {
                "annual_return": annual_return,
                "ages": ages,
                "liquid_series": liquid_series,
                "total_series": total_series,
                "min_total": min_total,
                "min_year": min_year,
                "hit_zero_year": hit_zero_year,
                "final_total": total_series[-1],
            }
        )

    return results


# =========================
# Core Monte Carlo model
# =========================
def run_monte_carlo_simulation(
    num_simulations: int,
    current_age: int,
    expected_age_of_death: int,
    retirement_access_age: int,
    monthly_income: float,
    monthly_expenses: float,
    non_retirement_savings: float,
    retirement_savings: float,
    inflation_rate: float,
    expense_reduction_age: int,
    expense_reduction_factor: float,
    stock_allocation: float,
    stock_mean: float,
    stock_std: float,
    bond_mean: float,
    bond_std: float,
    effective_tax_rate: float,
    legacy_goal: float,
):
    years = expected_age_of_death - current_age + 1
    bond_allocation = 1 - stock_allocation
    annual_income = monthly_income * 12

    def run_one_path():
        liquid_assets = non_retirement_savings
        retirement_assets = retirement_savings
        annual_expenses = monthly_expenses * 12

        path = []

        for year_idx in range(years):
            age = current_age + year_idx
            total_portfolio = liquid_assets + retirement_assets
            path.append(total_portfolio)

            if age == expected_age_of_death:
                break

            if age == retirement_access_age and retirement_assets > 0:
                liquid_assets += retirement_assets
                retirement_assets = 0

            if year_idx > 0:
                annual_expenses *= (1 + inflation_rate)

            if age == expense_reduction_age:
                annual_expenses *= expense_reduction_factor

            stock_return = np.random.normal(stock_mean, stock_std)
            bond_return = np.random.normal(bond_mean, bond_std)
            portfolio_return = (
                stock_allocation * stock_return + bond_allocation * bond_return
            )

            real_return = (1 + portfolio_return) / (1 + inflation_rate) - 1

            # Retirement assets compound until accessed
            if age < retirement_access_age and retirement_assets > 0:
                retirement_assets *= (1 + real_return)

            # Net spending from liquid assets
            gross_spending_need = max(annual_expenses - annual_income, 0)
            taxes = gross_spending_need * effective_tax_rate
            total_withdrawal = gross_spending_need + taxes

            liquid_assets -= total_withdrawal
            liquid_assets *= (1 + real_return)

        return path

    simulations = np.array([run_one_path() for _ in range(num_simulations)])
    ages = list(range(current_age, expected_age_of_death + 1))

    median_path = np.median(simulations, axis=0)
    percentile_10 = np.percentile(simulations, 10, axis=0)
    percentile_25 = np.percentile(simulations, 25, axis=0)
    percentile_75 = np.percentile(simulations, 75, axis=0)
    percentile_90 = np.percentile(simulations, 90, axis=0)

    final_values = simulations[:, -1]
    ruin_probability = np.mean(final_values <= 0)
    legacy_probability = np.mean(final_values >= legacy_goal)

    return {
        "ages": ages,
        "median_path": median_path,
        "percentile_10": percentile_10,
        "percentile_25": percentile_25,
        "percentile_75": percentile_75,
        "percentile_90": percentile_90,
        "final_values": final_values,
        "ruin_probability": ruin_probability,
        "legacy_probability": legacy_probability,
        "median_final": float(np.median(final_values)),
        "mean_final": float(np.mean(final_values)),
    }


# =========================
# Sidebar inputs
# =========================
st.title("Retirement Planning Calculator")
st.caption(
    "Explore deterministic return scenarios and Monte Carlo outcomes for retirement spending and portfolio sustainability."
)

with st.sidebar:
    st.header("Assumptions")

    st.subheader("Personal Timeline")
    current_age = st.number_input("Current age", min_value=18, max_value=100, value=56)
    retirement_access_age = st.number_input(
        "Retirement account access age",
        min_value=18,
        max_value=100,
        value=60,
        help="Age when retirement savings become available to support withdrawals.",
    )
    expected_age_of_death = st.number_input(
        "Planning age",
        min_value=50,
        max_value=120,
        value=95,
        help="Final age used in the projection.",
    )

    st.subheader("Current Assets & Cash Flow")
    non_retirement_savings = st.number_input(
        "Non-retirement savings ($)",
        min_value=0.0,
        value=500_000.0,
        step=10_000.0,
    )
    retirement_savings = st.number_input(
        "Retirement savings ($)",
        min_value=0.0,
        value=1_000_000.0,
        step=10_000.0,
    )
    monthly_income = st.number_input(
        "Monthly income ($)",
        min_value=0.0,
        value=500.0,
        step=100.0,
    )
    monthly_expenses = st.number_input(
        "Monthly expenses ($)",
        min_value=0.0,
        value=10_000.0,
        step=250.0,
    )

    st.subheader("Later-Life Spending Adjustment")
    expense_reduction_age = st.number_input(
        "Expense reduction age",
        min_value=18,
        max_value=120,
        value=80,
    )
    expense_reduction_factor = st.slider(
        "Expense reduction factor",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="0.50 means expenses are cut in half at the selected age.",
    )

    st.subheader("Deterministic Scenarios")
    selected_returns_pct = st.multiselect(
        "Annual return assumptions",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=[1, 2, 3, 4, 5],
        help="Used in the deterministic scenario analysis.",
    )
    annual_returns = [x / 100 for x in selected_returns_pct]

    st.subheader("Monte Carlo Assumptions")
    num_simulations = st.number_input(
        "Number of simulations",
        min_value=100,
        max_value=10000,
        value=500,
        step=100,
    )
    stock_allocation = st.slider(
        "Stock allocation",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.05,
    )
    inflation_rate = st.number_input(
        "Inflation rate",
        min_value=0.0,
        max_value=0.20,
        value=0.025,
        step=0.005,
        format="%.3f",
    )
    effective_tax_rate = st.number_input(
        "Effective tax rate on withdrawals",
        min_value=0.0,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f",
    )
    legacy_goal = st.number_input(
        "Legacy goal ($)",
        min_value=0.0,
        value=0.0,
        step=10_000.0,
    )

    stock_mean = st.number_input(
        "Stock mean return",
        min_value=-0.20,
        max_value=0.30,
        value=0.1,
        step=0.01,
        format="%.2f",
    )
    stock_std = st.number_input(
        "Stock volatility",
        min_value=0.0,
        max_value=0.50,
        value=0.15,
        step=0.01,
        format="%.2f",
    )
    bond_mean = st.number_input(
        "Bond mean return",
        min_value=-0.10,
        max_value=0.20,
        value=0.03,
        step=0.01,
        format="%.2f",
    )
    bond_std = st.number_input(
        "Bond volatility",
        min_value=0.0,
        max_value=0.30,
        value=0.05,
        step=0.01,
        format="%.2f",
    )

    run_analysis = st.button("Run analysis", type="primary", use_container_width=True)


# =========================
# Main app
# =========================
if not run_analysis:
    st.info("Set your assumptions in the sidebar, then click **Run analysis**.")
    st.stop()

if not annual_returns:
    st.error("Please select at least one deterministic return assumption.")
    st.stop()

det_results = run_deterministic_projection(
    annual_returns=annual_returns,
    monthly_income=monthly_income,
    monthly_expenses=monthly_expenses,
    non_retirement_savings=non_retirement_savings,
    retirement_savings=retirement_savings,
    current_age=int(current_age),
    retirement_growth_end_age=float(retirement_access_age),
    retirement_access_age=int(retirement_access_age),
    expected_age_of_death=int(expected_age_of_death),
    expense_reduction_age=int(expense_reduction_age),
    expense_reduction_factor=expense_reduction_factor,
)

mc_results = run_monte_carlo_simulation(
    num_simulations=int(num_simulations),
    current_age=int(current_age),
    expected_age_of_death=int(expected_age_of_death),
    retirement_access_age=int(retirement_access_age),
    monthly_income=monthly_income,
    monthly_expenses=monthly_expenses,
    non_retirement_savings=non_retirement_savings,
    retirement_savings=retirement_savings,
    inflation_rate=inflation_rate,
    expense_reduction_age=int(expense_reduction_age),
    expense_reduction_factor=expense_reduction_factor,
    stock_allocation=stock_allocation,
    stock_mean=stock_mean,
    stock_std=stock_std,
    bond_mean=bond_mean,
    bond_std=bond_std,
    effective_tax_rate=effective_tax_rate,
    legacy_goal=legacy_goal,
)

baseline_det = next((r for r in det_results if abs(r["annual_return"] - 0.05) < 1e-9), det_results[0])

# =========================
# Summary metrics
# =========================
m1, m2, m3, m4 = st.columns(4)

m1.metric("Starting portfolio", fmt_currency(non_retirement_savings + retirement_savings))
m2.metric("Baseline deterministic final value", fmt_currency(baseline_det["final_total"]))
m3.metric("Monte Carlo median final value", fmt_currency(mc_results["median_final"]))
m4.metric("Probability of meeting legacy goal", safe_percent(mc_results["legacy_probability"]))

st.markdown("---")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Overview", "Deterministic Scenarios", "Monte Carlo Simulation"])

with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Key Takeaways")

        ruin_text = safe_percent(mc_results["ruin_probability"])
        legacy_text = safe_percent(mc_results["legacy_probability"])
        baseline_zero = baseline_det["hit_zero_year"]

        st.write(
            f"""
            - The portfolio begins at **{fmt_currency(non_retirement_savings + retirement_savings)}**.
            - Under the selected baseline deterministic scenario, the projected ending portfolio value is **{fmt_currency(baseline_det['final_total'])}**.
            - In the Monte Carlo analysis, the **median** ending portfolio value is **{fmt_currency(mc_results['median_final'])}**.
            - The simulated probability of ending below zero by age {expected_age_of_death} is **{ruin_text}**.
            - The simulated probability of finishing at or above the selected legacy goal is **{legacy_text}**.
            """
        )

        if baseline_zero is not None:
            st.warning(f"In the baseline deterministic scenario, the portfolio falls below zero at age {baseline_zero}.")
        else:
            st.success("In the baseline deterministic scenario, the portfolio does not fall below zero.")

    with right:
        st.subheader("Assumption Snapshot")
        st.write(
            {
                "Current age": int(current_age),
                "Planning age": int(expected_age_of_death),
                "Retirement access age": int(retirement_access_age),
                "Monthly income": fmt_currency(monthly_income),
                "Monthly expenses": fmt_currency(monthly_expenses),
                "Expense reduction age": int(expense_reduction_age),
                "Expense reduction factor": f"{expense_reduction_factor:.0%}",
                "Inflation rate": safe_percent(inflation_rate),
                "Stock allocation": safe_percent(stock_allocation),
                "Simulations": int(num_simulations),
            }
        )

with tab2:
    st.subheader("Deterministic Return Scenarios")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for result in det_results:
        ax.plot(
            result["ages"],
            result["total_series"],
            linewidth=2,
            label=f"{result['annual_return']:.0%} annual return",
        )

    ax.set_title("Projected Portfolio Value Under Fixed Return Assumptions")
    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Value")
    ax.yaxis.set_major_formatter(FuncFormatter(currency_tick))
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Scenario Summary")

    for result in det_results:
        status = (
            f"Falls below zero at age {result['hit_zero_year']}"
            if result["hit_zero_year"] is not None
            else "Does not fall below zero"
        )
        st.write(
            f"**{result['annual_return']:.0%} return** — "
            f"Minimum portfolio value: {fmt_currency(result['min_total'])} at age {result['min_year']}; "
            f"Ending value: {fmt_currency(result['final_total'])}; "
            f"{status}."
        )

with tab3:
    st.subheader("Monte Carlo Simulation")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        mc_results["ages"],
        mc_results["median_path"],
        linewidth=2.25,
        label="Median outcome",
    )
    ax.fill_between(
        mc_results["ages"],
        mc_results["percentile_25"],
        mc_results["percentile_75"],
        alpha=0.35,
        label="25th–75th percentile range",
    )
    ax.fill_between(
        mc_results["ages"],
        mc_results["percentile_10"],
        mc_results["percentile_90"],
        alpha=0.18,
        label="10th–90th percentile range",
    )
    if legacy_goal > 0:
        ax.axhline(
            y=legacy_goal,
            linestyle="--",
            linewidth=1.5,
            label="Legacy goal",
        )

    ax.set_title("Projected Portfolio Distribution Under Simulated Returns")
    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Value")
    ax.yaxis.set_major_formatter(FuncFormatter(currency_tick))
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Median ending value", fmt_currency(mc_results["median_final"]))
    c2.metric("Mean ending value", fmt_currency(mc_results["mean_final"]))
    c3.metric("Probability of ruin", safe_percent(mc_results["ruin_probability"]))

    st.markdown("### Interpretation")
    st.write(
        """
        The Monte Carlo analysis models a range of possible market outcomes rather than assuming a fixed annual return.
        The median line shows the middle outcome across all simulations, while the shaded bands show how wide the
        distribution of outcomes becomes over time.
        """
    )

st.markdown("---")
st.caption(
    "This calculator is for educational and planning purposes only and should not be construed as personalized financial advice."
)
