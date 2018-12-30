use crate::utils::{match_type, paths, span_lint};
use rustc::hir::Ty;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty::subst::UnpackedKind;
use rustc::ty::TyKind;
use rustc::{declare_tool_lint, lint_array};

/// **What it does:** Checks for usage of `RandomState`
///
/// **Why is this bad?** Some applications don't need collision prevention
/// which lowers the performance.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn x() {
///     let mut map = std::collections::HashMap::new();
///     map.insert(3, 4);
/// }
/// ```
declare_clippy_lint! {
    pub RANDOM_STATE,
    nursery,
    "use of RandomState"
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(RANDOM_STATE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_ty(&mut self, cx: &LateContext<'a, 'tcx>, ty: &Ty) {
        if let TyKind::Adt(_, substs) = cx.tables.node_id_to_type(ty.hir_id).sty {
            for subst in substs {
                if let UnpackedKind::Type(build_hasher) = subst.unpack() {
                    if match_type(cx, build_hasher, &paths::RANDOM_STATE) {
                        span_lint(cx, RANDOM_STATE, ty.span, "usage of RandomState");
                    }
                }
            }
        }
    }
}
