use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_test;
use clippy_utils::msrvs::Msrv;
use rustc_attr_data_structures::{RustcVersion, Stability, StableSince};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{self as hir, AmbigArg, Expr, ExprKind, HirId, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::def_id::DefId;
use rustc_span::{ExpnKind, Span, sym};

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

pub struct IncompatibleMsrv {
    msrv: Msrv,
    availability_cache: FxHashMap<DefId, Availability>,
    check_in_tests: bool,
}

impl_lint_pass!(IncompatibleMsrv => [INCOMPATIBLE_MSRV]);

impl IncompatibleMsrv {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            availability_cache: FxHashMap::default(),
            check_in_tests: conf.check_incompatible_msrv_in_tests,
        }
    }

    /// Returns the availability of `def_id`, whether it is enabled through a feature or
    /// available since a given version (the default being Rust 1.0.0).
    fn get_def_id_availability(&mut self, tcx: TyCtxt<'_>, def_id: DefId) -> Availability {
        if let Some(availability) = self.availability_cache.get(&def_id) {
            return *availability;
        }
        let stability = tcx.lookup_stability(def_id);
        let version = if stability.is_some_and(|stability| tcx.features().enabled(stability.feature)) {
            Availability::FeatureEnabled
        } else if let Some(StableSince::Version(version)) = stability.as_ref().and_then(Stability::stable_since) {
            Availability::Since(version)
        } else if let Some(parent_def_id) = tcx.opt_parent(def_id) {
            self.get_def_id_availability(tcx, parent_def_id)
        } else {
            Availability::Since(RustcVersion {
                major: 1,
                minor: 0,
                patch: 0,
            })
        };
        self.availability_cache.insert(def_id, version);
        version
    }

    fn emit_lint_if_under_msrv(&mut self, cx: &LateContext<'_>, def_id: DefId, node: HirId, span: Span) {
        if def_id.is_local() {
            // We don't check local items since their MSRV is supposed to always be valid.
            return;
        }
        if let ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) = span.ctxt().outer_expn_data().kind {
            // Desugared expressions get to cheat and stability is ignored.
            // Intentionally not using `.from_expansion()`, since we do still care about macro expansions
            return;
        }

        // Functions coming from `core` while expanding a macro such as `assert*!()` get to cheat too: the
        // macros may have existed prior to the checked MSRV, but their expansion with a recent compiler
        // might use recent functions or methods. Compiling with an older compiler would not use those.
        if span.from_expansion()
            && cx.tcx.crate_name(def_id.krate) == sym::core
            && span
                .ctxt()
                .outer_expn_data()
                .macro_def_id
                .is_some_and(|def_id| cx.tcx.crate_name(def_id.krate) == sym::core)
        {
            return;
        }

        if (self.check_in_tests || !is_in_test(cx.tcx, node))
            && let Some(current) = self.msrv.current(cx)
            && let Availability::Since(version) = self.get_def_id_availability(cx.tcx, def_id)
            && version > current
        {
            span_lint_and_then(
                cx,
                INCOMPATIBLE_MSRV,
                span,
                format!(
                    "current MSRV (Minimum Supported Rust Version) is `{current}` but this item is stable since `{version}`"
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
        // TODO: check for const stability when in const context
        match expr.kind {
            ExprKind::MethodCall(_, _, _, span) => {
                if let Some(method_did) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    self.emit_lint_if_under_msrv(cx, method_did, expr.hir_id, span);
                }
            },
            // Desugaring into function calls by the compiler will use `QPath::LangItem` variants. Those should
            // not be linted as they will not be generated in older compilers if the function is not available,
            // and the compiler is allowed to call unstable functions.
            ExprKind::Path(qpath @ (QPath::Resolved(..) | QPath::TypeRelative(..))) => {
                if let Some(path_def_id) = cx.qpath_res(&qpath, expr.hir_id).opt_def_id() {
                    self.emit_lint_if_under_msrv(cx, path_def_id, expr.hir_id, expr.span);
                }
            },
            _ => {},
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, hir_ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
        if let hir::TyKind::Path(qpath @ (QPath::Resolved(..) | QPath::TypeRelative(..))) = hir_ty.kind
            && let Some(ty_def_id) = cx.qpath_res(&qpath, hir_ty.hir_id).opt_def_id()
            // `CStr` and `CString` have been moved around but have been available since Rust 1.0.0
            && !matches!(cx.tcx.get_diagnostic_name(ty_def_id), Some(sym::cstr_type | sym::cstring_type))
        {
            self.emit_lint_if_under_msrv(cx, ty_def_id, hir_ty.hir_id, hir_ty.span);
        }
    }
}

/// Heuristic checking if the node `hir_id` is under a `#[cfg()]` or `#[cfg_attr()]`
/// attribute.
fn is_under_cfg_attribute(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    cx.tcx.hir_parent_id_iter(hir_id).any(|id| {
        cx.tcx.hir_attrs(id).iter().any(|attr| {
            matches!(
                attr.ident().map(|ident| ident.name),
                Some(sym::cfg_trace | sym::cfg_attr_trace)
            )
        })
    })
}
