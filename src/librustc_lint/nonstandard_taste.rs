use crate::{LateContext, LateLintPass, LintContext};
use rustc_hir as hir;

declare_lint! {
    pub PINEAPPLE_ON_PIZZA,
    Forbid,
   "pineapple doesn't go on pizza"
}

declare_lint_pass!(PineappleOnPizza => [PINEAPPLE_ON_PIZZA]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PineappleOnPizza {
    fn check_ty(&mut self, cx: &LateContext<'_, '_>, ty: &hir::Ty<'_>) {
        if let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = &ty.kind {
            for pizza_segment in path.segments {
                if let Some(args) = pizza_segment.args {
                    if pizza_segment.ident.name.as_str().to_lowercase() != "pizza" {
                        continue;
                    }
                    for arg in args.args {
                        if let hir::GenericArg::Type(hir::Ty {
                            kind: hir::TyKind::Path(hir::QPath::Resolved(_, path)),
                            ..
                        }) = arg
                        {
                            for pineapple_segment in path.segments {
                                if pineapple_segment.ident.name.as_str().to_lowercase()
                                    == "pineapple"
                                {
                                    cx.struct_span_lint(
                                        PINEAPPLE_ON_PIZZA,
                                        pineapple_segment.ident.span,
                                        |lint| {
                                            let mut err =
                                                lint.build("pineapple doesn't go on pizza");
                                            err.span_label(
                                                pizza_segment.ident.span,
                                                "this is the pizza you ruined",
                                            );
                                            err.note("you're a monster"); // Yep Esteban, you are.
                                            err.emit();
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
