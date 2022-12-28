use clippy_utils::diagnostics::span_lint;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_inf, walk_ty, Visitor};
use rustc_hir::{GenericParamKind, TyKind};
use rustc_lint::LateContext;
use rustc_target::spec::abi::Abi;

use super::TYPE_COMPLEXITY;

pub(super) fn check(cx: &LateContext<'_>, ty: &hir::Ty<'_>, type_complexity_threshold: u64) -> bool {
    let score = {
        let mut visitor = TypeComplexityVisitor { score: 0, nest: 1 };
        visitor.visit_ty(ty);
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
    fn visit_infer(&mut self, inf: &'tcx hir::InferArg) {
        self.score += 1;
        walk_inf(self, inf);
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'_>) {
        let (add_score, sub_nest) = match ty.kind {
            // _, &x and *x have only small overhead; don't mess with nesting level
            TyKind::Infer | TyKind::Ptr(..) | TyKind::Ref(..) => (1, 0),

            // the "normal" components of a type: named types, arrays/tuples
            TyKind::Path(..) | TyKind::Slice(..) | TyKind::Tup(..) | TyKind::Array(..) => (10 * self.nest, 1),

            // function types bring a lot of overhead
            TyKind::BareFn(bare) if bare.abi == Abi::Rust => (50 * self.nest, 1),

            TyKind::TraitObject(param_bounds, _, _) => {
                let has_lifetime_parameters = param_bounds.iter().any(|bound| {
                    bound
                        .bound_generic_params
                        .iter()
                        .any(|gen| matches!(gen.kind, GenericParamKind::Lifetime { .. }))
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
