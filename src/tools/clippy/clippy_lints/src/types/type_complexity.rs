use clippy_utils::diagnostics::span_lint;
use rustc_abi::ExternAbi;
use rustc_hir::intravisit::{InferKind, Visitor, VisitorExt, walk_ty};
use rustc_hir::{self as hir, AmbigArg, GenericParamKind, TyKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::TYPE_COMPLEXITY;

pub(super) fn check(cx: &LateContext<'_>, ty: &hir::Ty<'_>, type_complexity_threshold: u64) -> bool {
    let score = {
        let mut visitor = TypeComplexityVisitor { score: 0, nest: 1 };
        visitor.visit_ty_unambig(ty);
        visitor.score
    };

    if score > type_complexity_threshold {
        span_lint(
            cx,
            TYPE_COMPLEXITY,
            ty.span,
            "very complex type used. Consider factoring parts into `type` definitions",
        );
        true
    } else {
        false
    }
}

/// Walks a type and assigns a complexity score to it.
struct TypeComplexityVisitor {
    /// total complexity score of the type
    score: u64,
    /// current nesting level
    nest: u64,
}

impl<'tcx> Visitor<'tcx> for TypeComplexityVisitor {
    fn visit_infer(&mut self, inf_id: hir::HirId, _inf_span: Span, _kind: InferKind<'tcx>) -> Self::Result {
        self.score += 1;
        self.visit_id(inf_id);
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'_, AmbigArg>) {
        let (add_score, sub_nest) = match ty.kind {
            // &x and *x have only small overhead; don't mess with nesting level
            TyKind::Ptr(..) | TyKind::Ref(..) => (1, 0),

            // the "normal" components of a type: named types, arrays/tuples
            TyKind::Path(..) | TyKind::Slice(..) | TyKind::Tup(..) | TyKind::Array(..) => (10 * self.nest, 1),

            // function types bring a lot of overhead
            TyKind::FnPtr(fn_ptr) if fn_ptr.abi == ExternAbi::Rust => (50 * self.nest, 1),

            TyKind::TraitObject(param_bounds, _) => {
                let has_lifetime_parameters = param_bounds.iter().any(|bound| {
                    bound
                        .bound_generic_params
                        .iter()
                        .any(|param| matches!(param.kind, GenericParamKind::Lifetime { .. }))
                });
                if has_lifetime_parameters {
                    // complex trait bounds like A<'a, 'b>
                    (50 * self.nest, 1)
                } else {
                    // simple trait bounds like A + B
                    (20 * self.nest, 0)
                }
            },

            _ => (0, 0),
        };
        self.score += add_score;
        self.nest += sub_nest;
        walk_ty(self, ty);
        self.nest -= sub_nest;
    }
}
