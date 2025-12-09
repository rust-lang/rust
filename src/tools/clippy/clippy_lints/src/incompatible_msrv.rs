use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::Msrv;
use clippy_utils::{is_in_const_context, is_in_test, sym};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{self as hir, AmbigArg, Expr, ExprKind, HirId, RustcVersion, StabilityLevel, StableSince};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::def_id::{CrateNum, DefId};
use rustc_span::{ExpnKind, Span};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// This lint checks that no function newer than the defined MSRV (minimum
    /// supported rust version) is used in the crate.
    ///
    /// ### Why is this bad?
    ///
    /// It would prevent the crate to be actually used with the specified MSRV.
    ///
    /// ### Example
    /// ```no_run
    /// // MSRV of 1.3.0
    /// use std::thread::sleep;
    /// use std::time::Duration;
    ///
    /// // Sleep was defined in `1.4.0`.
    /// sleep(Duration::new(1, 0));
    /// ```
    ///
    /// To fix this problem, either increase your MSRV or use another item
    /// available in your current MSRV.
    ///
    /// You can also locally change the MSRV that should be checked by Clippy,
    /// for example if a feature in your crate (e.g., `modern_compiler`) should
    /// allow you to use an item:
    ///
    /// ```no_run
    /// //! This crate has a MSRV of 1.3.0, but we also have an optional feature
    /// //! `sleep_well` which requires at least Rust 1.4.0.
    ///
    /// // When the `sleep_well` feature is set, do not warn for functions available
    /// // in Rust 1.4.0 and below.
    /// #![cfg_attr(feature = "sleep_well", clippy::msrv = "1.4.0")]
    ///
    /// use std::time::Duration;
    ///
    /// #[cfg(feature = "sleep_well")]
    /// fn sleep_for_some_time() {
    ///     std::thread::sleep(Duration::new(1, 0)); // Will not trigger the lint
    /// }
    /// ```
    ///
    /// You can also increase the MSRV in tests, by using:
    ///
    /// ```no_run
    /// // Use a much higher MSRV for tests while keeping the main one low
    /// #![cfg_attr(test, clippy::msrv = "1.85.0")]
    ///
    /// #[test]
    /// fn my_test() {
    ///     // The tests can use items introduced in Rust 1.85.0 and lower
    ///     // without triggering the `incompatible_msrv` lint.
    /// }
    /// ```
    #[clippy::version = "1.78.0"]
    pub INCOMPATIBLE_MSRV,
    suspicious,
    "ensures that all items used in the crate are available for the current MSRV"
}

#[derive(Clone, Copy)]
enum Availability {
    FeatureEnabled,
    Since(RustcVersion),
}

/// All known std crates containing a stability attribute.
struct StdCrates([Option<CrateNum>; 6]);
impl StdCrates {
    fn new(tcx: TyCtxt<'_>) -> Self {
        let mut res = Self([None; _]);
        for &krate in tcx.crates(()) {
            // FIXME(@Jarcho): We should have an internal lint to detect when this list is out of date.
            match tcx.crate_name(krate) {
                sym::alloc => res.0[0] = Some(krate),
                sym::core => res.0[1] = Some(krate),
                sym::core_arch => res.0[2] = Some(krate),
                sym::proc_macro => res.0[3] = Some(krate),
                sym::std => res.0[4] = Some(krate),
                sym::std_detect => res.0[5] = Some(krate),
                _ => {},
            }
        }
        res
    }

    fn contains(&self, krate: CrateNum) -> bool {
        self.0.contains(&Some(krate))
    }
}

pub struct IncompatibleMsrv {
    msrv: Msrv,
    availability_cache: FxHashMap<(DefId, bool), Availability>,
    check_in_tests: bool,
    std_crates: StdCrates,

    // The most recently called path. Used to skip checking the path after it's
    // been checked when visiting the call expression.
    called_path: Option<HirId>,
}

impl_lint_pass!(IncompatibleMsrv => [INCOMPATIBLE_MSRV]);

impl IncompatibleMsrv {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            availability_cache: FxHashMap::default(),
            check_in_tests: conf.check_incompatible_msrv_in_tests,
            std_crates: StdCrates::new(tcx),
            called_path: None,
        }
    }

    /// Returns the availability of `def_id`, whether it is enabled through a feature or
    /// available since a given version (the default being Rust 1.0.0). `needs_const` requires
    /// the `const`-stability to be looked up instead of the stability in non-`const` contexts.
    fn get_def_id_availability(&mut self, tcx: TyCtxt<'_>, def_id: DefId, needs_const: bool) -> Availability {
        if let Some(availability) = self.availability_cache.get(&(def_id, needs_const)) {
            return *availability;
        }
        let (feature, stability_level) = if needs_const {
            tcx.lookup_const_stability(def_id)
                .map(|stability| (stability.feature, stability.level))
                .unzip()
        } else {
            tcx.lookup_stability(def_id)
                .map(|stability| (stability.feature, stability.level))
                .unzip()
        };
        let version = if feature.is_some_and(|feature| tcx.features().enabled(feature)) {
            Availability::FeatureEnabled
        } else if let Some(StableSince::Version(version)) =
            stability_level.as_ref().and_then(StabilityLevel::stable_since)
        {
            Availability::Since(version)
        } else if needs_const {
            // Fallback to regular stability
            self.get_def_id_availability(tcx, def_id, false)
        } else if let Some(parent_def_id) = tcx.opt_parent(def_id) {
            self.get_def_id_availability(tcx, parent_def_id, needs_const)
        } else {
            Availability::Since(RustcVersion {
                major: 1,
                minor: 0,
                patch: 0,
            })
        };
        self.availability_cache.insert((def_id, needs_const), version);
        version
    }

    /// Emit lint if `def_id`, associated with `node` and `span`, is below the current MSRV.
    fn emit_lint_if_under_msrv(
        &mut self,
        cx: &LateContext<'_>,
        needs_const: bool,
        def_id: DefId,
        node: HirId,
        span: Span,
    ) {
        if !self.std_crates.contains(def_id.krate) {
            // No stability attributes to lookup for these items.
            return;
        }
        // Use `from_expansion` to fast-path the common case.
        if span.from_expansion() {
            let expn = span.ctxt().outer_expn_data();
            match expn.kind {
                // FIXME(@Jarcho): Check that the actual desugaring or std macro is supported by the
                // current MSRV. Note that nested expansions need to be handled as well.
                ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => return,
                ExpnKind::Macro(..) if expn.macro_def_id.is_some_and(|did| self.std_crates.contains(did.krate)) => {
                    return;
                },
                // All other expansions share the target's MSRV.
                // FIXME(@Jarcho): What should we do about version dependant macros from external crates?
                _ => {},
            }
        }

        if (self.check_in_tests || !is_in_test(cx.tcx, node))
            && let Some(current) = self.msrv.current(cx)
            && let Availability::Since(version) = self.get_def_id_availability(cx.tcx, def_id, needs_const)
            && version > current
        {
            span_lint_and_then(
                cx,
                INCOMPATIBLE_MSRV,
                span,
                format!(
                    "current MSRV (Minimum Supported Rust Version) is `{current}` but this item is stable{} since `{version}`",
                    if needs_const { " in a `const` context" } else { "" },
                ),
                |diag| {
                    if is_under_cfg_attribute(cx, node) {
                        diag.note_once("you may want to conditionally increase the MSRV considered by Clippy using the `clippy::msrv` attribute");
                    }
                },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for IncompatibleMsrv {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        match expr.kind {
            ExprKind::MethodCall(_, _, _, span) => {
                if let Some(method_did) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    self.emit_lint_if_under_msrv(cx, is_in_const_context(cx), method_did, expr.hir_id, span);
                }
            },
            ExprKind::Call(callee, _) if let ExprKind::Path(qpath) = callee.kind => {
                self.called_path = Some(callee.hir_id);
                let needs_const = is_in_const_context(cx);
                let def_id = if let Some(def_id) = cx.qpath_res(&qpath, callee.hir_id).opt_def_id() {
                    def_id
                } else if needs_const && let ty::FnDef(def_id, _) = *cx.typeck_results().expr_ty(callee).kind() {
                    // Edge case where a function is first assigned then called.
                    // We previously would have warned for the non-const MSRV, when
                    // checking the path, but now that it's called the const MSRV
                    // must also be met.
                    def_id
                } else {
                    return;
                };
                self.emit_lint_if_under_msrv(cx, needs_const, def_id, expr.hir_id, callee.span);
            },
            // Desugaring into function calls by the compiler will use `QPath::LangItem` variants. Those should
            // not be linted as they will not be generated in older compilers if the function is not available,
            // and the compiler is allowed to call unstable functions.
            ExprKind::Path(qpath)
                if let Some(path_def_id) = cx.qpath_res(&qpath, expr.hir_id).opt_def_id()
                    && self.called_path != Some(expr.hir_id) =>
            {
                self.emit_lint_if_under_msrv(cx, false, path_def_id, expr.hir_id, expr.span);
            },
            _ => {},
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, hir_ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
        if let hir::TyKind::Path(qpath) = hir_ty.kind
            && let Some(ty_def_id) = cx.qpath_res(&qpath, hir_ty.hir_id).opt_def_id()
            // `CStr` and `CString` have been moved around but have been available since Rust 1.0.0
            && !matches!(cx.tcx.get_diagnostic_name(ty_def_id), Some(sym::cstr_type | sym::cstring_type))
        {
            self.emit_lint_if_under_msrv(cx, false, ty_def_id, hir_ty.hir_id, hir_ty.span);
        }
    }
}

/// Heuristic checking if the node `hir_id` is under a `#[cfg()]` or `#[cfg_attr()]`
/// attribute.
fn is_under_cfg_attribute(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    cx.tcx.hir_parent_id_iter(hir_id).any(|id| {
        cx.tcx.hir_attrs(id).iter().any(|attr| {
            matches!(
                attr.name(),
                Some(sym::cfg_trace | sym::cfg_attr_trace)
            )
        })
    })
}
