use std::ops::ControlFlow;

use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath, RustcVersion};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::Symbol;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for instances where a `std::time::Duration` is constructed using a smaller time unit
    /// when the value could be expressed more clearly using a larger unit.
    ///
    /// ### Why is this bad?
    ///
    /// Using a smaller unit for a duration that is evenly divisible by a larger unit reduces
    /// readability. Readers have to mentally convert values, which can be error-prone and makes
    /// the code less clear.
    ///
    /// ### Example
    /// ```
    /// use std::time::Duration;
    ///
    /// let dur = Duration::from_millis(5_000);
    /// let dur = Duration::from_secs(180);
    /// let dur = Duration::from_mins(10 * 60);
    /// ```
    ///
    /// Use instead:
    /// ```
    /// use std::time::Duration;
    ///
    /// let dur = Duration::from_secs(5);
    /// let dur = Duration::from_mins(3);
    /// let dur = Duration::from_hours(10);
    /// ```
    #[clippy::version = "1.95.0"]
    pub DURATION_SUBOPTIMAL_UNITS,
    pedantic,
    "constructing a `Duration` using a smaller unit when a larger unit would be more readable"
}

impl_lint_pass!(DurationSuboptimalUnits => [DURATION_SUBOPTIMAL_UNITS]);

pub struct DurationSuboptimalUnits {
    msrv: Msrv,
    units: Vec<Unit>,
}

impl DurationSuboptimalUnits {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        // The order of the units matters, as they are walked top to bottom
        let mut units = UNITS.to_vec();
        if tcx.features().enabled(sym::duration_constructors) {
            units.extend(EXTENDED_UNITS);
        }
        Self { msrv: conf.msrv, units }
    }
}

impl LateLintPass<'_> for DurationSuboptimalUnits {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if !expr.span.in_external_macro(cx.sess().source_map())
            // Check if a function on std::time::Duration is called
            && let ExprKind::Call(func, [arg]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(func_ty, func_name)) = func.kind
            && cx
                .typeck_results()
                .node_type(func_ty.hir_id)
                .is_diag_item(cx, sym::Duration)
            // We intentionally don't want to evaluate referenced constants, as we don't want to
            // recommend a literal value over using constants:
            //
            // let dur = Duration::from_secs(SIXTY);
            //           ^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `Duration::from_mins(1)`
            && let Some(Constant::Int(value)) = ConstEvalCtxt::new(cx).eval_local(arg, expr.span.ctxt())
            && let value = u64::try_from(value).expect("All Duration::from_<time-unit> constructors take a u64")
            // There is no need to promote e.g. 0 seconds to 0 hours
            && value != 0
            && let Some((promoted_constructor, promoted_value)) = self.promote(cx, func_name.ident.name, value)
        {
            span_lint_and_then(
                cx,
                DURATION_SUBOPTIMAL_UNITS,
                expr.span,
                "constructing a `Duration` using a smaller unit when a larger unit would be more readable",
                |diag| {
                    let suggestions = vec![
                        (func_name.ident.span, promoted_constructor.to_string()),
                        (arg.span, promoted_value.to_string()),
                    ];
                    diag.multipart_suggestion_verbose(
                        format!("try using {promoted_constructor}"),
                        suggestions,
                        Applicability::MachineApplicable,
                    );
                },
            );
        }
    }
}

impl DurationSuboptimalUnits {
    /// Tries to promote the given constructor and value to a bigger time unit and returns the
    /// promoted constructor name and value.
    ///
    /// Returns [`None`] in case no promotion could be done.
    fn promote(&self, cx: &LateContext<'_>, constructor_name: Symbol, value: u64) -> Option<(Symbol, u64)> {
        let (best_unit, best_value) = self
            .units
            .iter()
            .skip_while(|unit| unit.constructor_name != constructor_name)
            .skip(1)
            .try_fold(
                (constructor_name, value),
                |(current_unit, current_value), bigger_unit| {
                    if let Some(bigger_value) = current_value.div_exact(u64::from(bigger_unit.factor))
                        && bigger_unit.stable_since.is_none_or(|v| self.msrv.meets(cx, v))
                    {
                        ControlFlow::Continue((bigger_unit.constructor_name, bigger_value))
                    } else {
                        // We have to break early, as we can't skip versions, as they are needed to
                        // correctly calculate the promoted value.
                        ControlFlow::Break((current_unit, current_value))
                    }
                },
            )
            .into_value();
        (best_unit != constructor_name).then_some((best_unit, best_value))
    }
}

#[derive(Clone, Copy)]
struct Unit {
    /// Name of the constructor on [`Duration`](std::time::Duration) to construct it from the given
    /// unit, e.g. [`Duration::from_secs`](std::time::Duration::from_secs)
    constructor_name: Symbol,

    /// The increase factor over the previous (smaller) unit
    factor: u16,

    /// In what rustc version stable support for this constructor was added.
    /// We do not need to track the version stable support in const contexts was added, as the const
    /// stabilization was done in an ascending order of the time unites, so it's always valid to
    /// promote a const constructor.
    stable_since: Option<RustcVersion>,
}

/// Time unit constructors available on stable. The order matters!
const UNITS: [Unit; 6] = [
    Unit {
        constructor_name: sym::from_nanos,
        // The value doesn't matter, as there is no previous unit
        factor: 0,
        stable_since: Some(msrvs::DURATION_FROM_NANOS_MICROS),
    },
    Unit {
        constructor_name: sym::from_micros,
        factor: 1_000,
        stable_since: Some(msrvs::DURATION_FROM_NANOS_MICROS),
    },
    Unit {
        constructor_name: sym::from_millis,
        factor: 1_000,
        stable_since: Some(msrvs::DURATION_FROM_MILLIS_SECS),
    },
    Unit {
        constructor_name: sym::from_secs,
        factor: 1_000,
        stable_since: Some(msrvs::DURATION_FROM_MILLIS_SECS),
    },
    Unit {
        constructor_name: sym::from_mins,
        factor: 60,
        stable_since: Some(msrvs::DURATION_FROM_MINUTES_HOURS),
    },
    Unit {
        constructor_name: sym::from_hours,
        factor: 60,
        stable_since: Some(msrvs::DURATION_FROM_MINUTES_HOURS),
    },
];

/// Time unit constructors behind the `duration_constructors` feature. The order matters!
const EXTENDED_UNITS: [Unit; 2] = [
    Unit {
        constructor_name: sym::from_days,
        factor: 24,
        stable_since: None,
    },
    Unit {
        constructor_name: sym::from_weeks,
        factor: 7,
        stable_since: None,
    },
];
