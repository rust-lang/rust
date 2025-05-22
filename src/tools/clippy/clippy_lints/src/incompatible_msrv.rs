use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_in_test;
use clippy_utils::msrvs::Msrv;
use rustc_attr_data_structures::{RustcVersion, StabilityLevel, StableSince};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{Expr, ExprKind, HirId, QPath};
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
    #[clippy::version = "1.78.0"]
    pub INCOMPATIBLE_MSRV,
    suspicious,
    "ensures that all items used in the crate are available for the current MSRV"
}

pub struct IncompatibleMsrv {
    msrv: Msrv,
    is_above_msrv: FxHashMap<DefId, RustcVersion>,
    check_in_tests: bool,
}

impl_lint_pass!(IncompatibleMsrv => [INCOMPATIBLE_MSRV]);

impl IncompatibleMsrv {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            is_above_msrv: FxHashMap::default(),
            check_in_tests: conf.check_incompatible_msrv_in_tests,
        }
    }

    fn get_def_id_version(&mut self, tcx: TyCtxt<'_>, def_id: DefId) -> RustcVersion {
        if let Some(version) = self.is_above_msrv.get(&def_id) {
            return *version;
        }
        let version = if let Some(version) = tcx
            .lookup_stability(def_id)
            .and_then(|stability| match stability.level {
                StabilityLevel::Stable {
                    since: StableSince::Version(version),
                    ..
                } => Some(version),
                _ => None,
            }) {
            version
        } else if let Some(parent_def_id) = tcx.opt_parent(def_id) {
            self.get_def_id_version(tcx, parent_def_id)
        } else {
            RustcVersion {
                major: 1,
                minor: 0,
                patch: 0,
            }
        };
        self.is_above_msrv.insert(def_id, version);
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
            && let version = self.get_def_id_version(cx.tcx, def_id)
            && version > current
        {
            span_lint(
                cx,
                INCOMPATIBLE_MSRV,
                span,
                format!(
                    "current MSRV (Minimum Supported Rust Version) is `{current}` but this item is stable since `{version}`"
                ),
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for IncompatibleMsrv {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        match expr.kind {
            ExprKind::MethodCall(_, _, _, span) => {
                if let Some(method_did) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    self.emit_lint_if_under_msrv(cx, method_did, expr.hir_id, span);
                }
            },
            ExprKind::Call(call, _) => {
                // Desugaring into function calls by the compiler will use `QPath::LangItem` variants. Those should
                // not be linted as they will not be generated in older compilers if the function is not available,
                // and the compiler is allowed to call unstable functions.
                if let ExprKind::Path(qpath @ (QPath::Resolved(..) | QPath::TypeRelative(..))) = call.kind
                    && let Some(path_def_id) = cx.qpath_res(&qpath, call.hir_id).opt_def_id()
                {
                    self.emit_lint_if_under_msrv(cx, path_def_id, expr.hir_id, call.span);
                }
            },
            _ => {},
        }
    }
}
