use clippy_config::msrvs::Msrv;
use clippy_utils::diagnostics::span_lint;
use rustc_attr::{StabilityLevel, StableSince};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_semver::RustcVersion;
use rustc_session::impl_lint_pass;
use rustc_span::def_id::DefId;
use rustc_span::Span;

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
    #[clippy::version = "1.77.0"]
    pub INCOMPATIBLE_MSRV,
    suspicious,
    "ensures that all items used in the crate are available for the current MSRV"
}

pub struct IncompatibleMsrv {
    msrv: Msrv,
    is_above_msrv: FxHashMap<DefId, RustcVersion>,
}

impl_lint_pass!(IncompatibleMsrv => [INCOMPATIBLE_MSRV]);

impl IncompatibleMsrv {
    pub fn new(msrv: Msrv) -> Self {
        Self {
            msrv,
            is_above_msrv: FxHashMap::default(),
        }
    }

    #[allow(clippy::cast_lossless)]
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
                } => Some(RustcVersion::new(
                    version.major as _,
                    version.minor as _,
                    version.patch as _,
                )),
                _ => None,
            }) {
            version
        } else if let Some(parent_def_id) = tcx.opt_parent(def_id) {
            self.get_def_id_version(tcx, parent_def_id)
        } else {
            RustcVersion::new(1, 0, 0)
        };
        self.is_above_msrv.insert(def_id, version);
        version
    }

    fn emit_lint_if_under_msrv(&mut self, cx: &LateContext<'_>, def_id: DefId, span: Span) {
        if def_id.is_local() {
            // We don't check local items since their MSRV is supposed to always be valid.
            return;
        }
        let version = self.get_def_id_version(cx.tcx, def_id);
        if self.msrv.meets(version) {
            return;
        }
        self.emit_lint_for(cx, span, version);
    }

    fn emit_lint_for(&self, cx: &LateContext<'_>, span: Span, version: RustcVersion) {
        span_lint(
            cx,
            INCOMPATIBLE_MSRV,
            span,
            &format!(
                "current MSRV (Minimum Supported Rust Version) is `{}` but this item is stable since `{version}`",
                self.msrv
            ),
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for IncompatibleMsrv {
    extract_msrv_attr!(LateContext);

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if self.msrv.current().is_none() {
            // If there is no MSRV, then no need to check anything...
            return;
        }
        match expr.kind {
            ExprKind::MethodCall(_, _, _, span) => {
                if let Some(method_did) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    self.emit_lint_if_under_msrv(cx, method_did, span);
                }
            },
            ExprKind::Call(call, [_]) => {
                if let ExprKind::Path(qpath) = call.kind
                    && let Some(path_def_id) = cx.qpath_res(&qpath, call.hir_id).opt_def_id()
                {
                    self.emit_lint_if_under_msrv(cx, path_def_id, call.span);
                }
            },
            _ => {},
        }
    }
}
