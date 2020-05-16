use crate::LoweringContext;
use rustc_ast::ast::{
    AttrVec, Expr, ExprKind, GenericBound, PolyTraitRef, TraitBoundModifier, TraitObjectSyntax,
    TraitRef, Ty, TyKind,
};
use rustc_ast::ptr::P;
use rustc_errors::Applicability;
use rustc_hir::def::Namespace;
use rustc_hir::definitions::DefPathData;
use rustc_hir::{AnonConst, ConstArg, GenericArg};
use rustc_span::hygiene::ExpnId;

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    /// Possible `a + b` expression that should be surrounded in braces but was parsed
    /// as trait bounds in a trait object. Suggest surrounding with braces.
    crate fn detect_const_expr_as_trait_object(&mut self, ty: &P<Ty>) -> Option<GenericArg<'hir>> {
        if let TyKind::TraitObject(ref bounds, TraitObjectSyntax::None) = ty.kind {
            // We cannot disambiguate multi-segment paths right now as that requires type
            // checking.
            let const_expr_without_braces = bounds.iter().all(|bound| match bound {
                GenericBound::Trait(
                    PolyTraitRef { bound_generic_params, trait_ref: TraitRef { path, .. }, .. },
                    TraitBoundModifier::None,
                ) if bound_generic_params.is_empty()
                    && path.segments.len() == 1
                    && path.segments[0].args.is_none() =>
                {
                    let part_res = self.resolver.get_partial_res(path.segments[0].id);
                    match part_res.map(|r| r.base_res()) {
                        Some(res) => {
                            !res.matches_ns(Namespace::TypeNS) && res.matches_ns(Namespace::ValueNS)
                        }
                        None => true,
                    }
                }
                _ => false,
            });
            if const_expr_without_braces {
                self.sess.struct_span_err(ty.span, "likely `const` expression parsed as trait bounds")
                    .span_label(ty.span, "parsed as trait bounds but traits weren't found")
                    .multipart_suggestion(
                        "if you meant to write a `const` expression, surround the expression with braces",
                        vec![
                            (ty.span.shrink_to_lo(), "{ ".to_string()),
                            (ty.span.shrink_to_hi(), " }".to_string()),
                        ],
                        Applicability::MachineApplicable,
                    )
                    .emit();

                let parent_def_id = self.current_hir_id_owner.last().unwrap().0;
                let node_id = self.resolver.next_node_id();
                // Add a definition for the in-band const def.
                self.resolver.definitions().create_def_with_parent(
                    parent_def_id,
                    node_id,
                    DefPathData::AnonConst,
                    ExpnId::root(),
                    ty.span,
                );

                let path_expr =
                    Expr { id: ty.id, kind: ExprKind::Err, span: ty.span, attrs: AttrVec::new() };
                let value = self.with_new_scopes(|this| AnonConst {
                    hir_id: this.lower_node_id(node_id),
                    body: this.lower_const_body(path_expr.span, Some(&path_expr)),
                });
                return Some(GenericArg::Const(ConstArg { value, span: ty.span }));
            }
        }
        None
    }
}
