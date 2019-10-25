use crate::const_eval::const_variant_index;

use rustc::hir;
use rustc::lint;
use rustc::mir::Field;
use rustc::traits::{ObligationCause, PredicateObligation};
use rustc::ty;

use rustc_index::vec::Idx;

use syntax::symbol::sym;
use syntax_pos::Span;

use super::{FieldPat, Pat, PatCtxt, PatKind};

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
    /// Converts an evaluated constant to a pattern (if possible).
    /// This means aggregate values (like structs and enums) are converted
    /// to a pattern that matches the value (as if you'd compared via structural equality).
    pub(super) fn const_to_pat(
        &self,
        instance: ty::Instance<'tcx>,
        cv: &'tcx ty::Const<'tcx>,
        id: hir::HirId,
        span: Span,
    ) -> Pat<'tcx> {
        // This method is just a warpper handling a validity check; the heavy lifting is
        // performed by the recursive const_to_pat_inner method, which is not meant to be
        // invoked except by this method.
        //
        // once indirect_structural_match is a full fledged error, this
        // level of indirection can be eliminated

        debug!("const_to_pat: cv={:#?} id={:?}", cv, id);
        debug!("const_to_pat: cv.ty={:?} span={:?}", cv.ty, span);

        let mut saw_error = false;
        let inlined_const_as_pat = self.const_to_pat_inner(instance, cv, id, span, &mut saw_error);

        if self.include_lint_checks && !saw_error {
            // If we were able to successfully convert the const to some pat, double-check
            // that the type of the const obeys `#[structural_match]` constraint.
            if let Some(non_sm_ty) = ty::search_for_structural_match_violation(self.tcx, cv.ty) {
                let msg = match non_sm_ty {
                    ty::NonStructuralMatchTy::Adt(adt_def) => {
                        let path = self.tcx.def_path_str(adt_def.did);
                        format!(
                            "to use a constant of type `{}` in a pattern, \
                             `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                            path,
                            path,
                        )
                    }
                    ty::NonStructuralMatchTy::Param => {
                        bug!("use of constant whose type is a parameter inside a pattern");
                    }
                };

                // before issuing lint, double-check there even *is* a
                // semantic PartialEq for us to dispatch to.
                //
                // (If there isn't, then we can safely issue a hard
                // error, because that's never worked, due to compiler
                // using PartialEq::eq in this scenario in the past.)

                let ty_is_partial_eq: bool = {
                    let partial_eq_trait_id = self.tcx.lang_items().eq_trait().unwrap();
                    let obligation: PredicateObligation<'_> =
                        self.tcx.predicate_for_trait_def(self.param_env,
                                                         ObligationCause::misc(span, id),
                                                         partial_eq_trait_id,
                                                         0,
                                                         cv.ty,
                                                         &[]);
                    self.tcx
                        .infer_ctxt()
                        .enter(|infcx| infcx.predicate_may_hold(&obligation))
                };

                if !ty_is_partial_eq {
                    // span_fatal avoids ICE from resolution of non-existent method (rare case).
                    self.tcx.sess.span_fatal(span, &msg);
                } else {
                    self.tcx.lint_hir(lint::builtin::INDIRECT_STRUCTURAL_MATCH, id, span, &msg);
                }
            }
        }

        inlined_const_as_pat
    }

    /// Recursive helper for `const_to_pat`; invoke that (instead of calling this directly).
    fn const_to_pat_inner(
        &self,
        instance: ty::Instance<'tcx>,
        cv: &'tcx ty::Const<'tcx>,
        id: hir::HirId,
        span: Span,
        // This tracks if we signal some hard error for a given const
        // value, so that we will not subsequently issue an irrelevant
        // lint for the same const value.
        saw_const_match_error: &mut bool,
    ) -> Pat<'tcx> {

        let mut adt_subpattern = |i, variant_opt| {
            let field = Field::new(i);
            let val = crate::const_eval::const_field(
                self.tcx, self.param_env, variant_opt, field, cv
            );
            self.const_to_pat_inner(instance, val, id, span, saw_const_match_error)
        };
        let mut adt_subpatterns = |n, variant_opt| {
            (0..n).map(|i| {
                let field = Field::new(i);
                FieldPat {
                    field,
                    pattern: adt_subpattern(i, variant_opt),
                }
            }).collect::<Vec<_>>()
        };


        let kind = match cv.ty.kind {
            ty::Float(_) => {
                self.tcx.lint_hir(
                    ::rustc::lint::builtin::ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
                    id,
                    span,
                    "floating-point types cannot be used in patterns",
                );
                PatKind::Constant {
                    value: cv,
                }
            }
            ty::Adt(adt_def, _) if adt_def.is_union() => {
                // Matching on union fields is unsafe, we can't hide it in constants
                *saw_const_match_error = true;
                self.tcx.sess.span_err(span, "cannot use unions in constant patterns");
                PatKind::Wild
            }
            // keep old code until future-compat upgraded to errors.
            ty::Adt(adt_def, _) if !self.tcx.has_attr(adt_def.did, sym::structural_match) => {
                let path = self.tcx.def_path_str(adt_def.did);
                let msg = format!(
                    "to use a constant of type `{}` in a pattern, \
                     `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                    path,
                    path,
                );
                *saw_const_match_error = true;
                self.tcx.sess.span_err(span, &msg);
                PatKind::Wild
            }
            // keep old code until future-compat upgraded to errors.
            ty::Ref(_, ty::TyS { kind: ty::Adt(adt_def, _), .. }, _)
            if !self.tcx.has_attr(adt_def.did, sym::structural_match) => {
                // HACK(estebank): Side-step ICE #53708, but anything other than erroring here
                // would be wrong. Returnging `PatKind::Wild` is not technically correct.
                let path = self.tcx.def_path_str(adt_def.did);
                let msg = format!(
                    "to use a constant of type `{}` in a pattern, \
                     `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                    path,
                    path,
                );
                *saw_const_match_error = true;
                self.tcx.sess.span_err(span, &msg);
                PatKind::Wild
            }
            ty::Adt(adt_def, substs) if adt_def.is_enum() => {
                let variant_index = const_variant_index(self.tcx, self.param_env, cv);
                let subpatterns = adt_subpatterns(
                    adt_def.variants[variant_index].fields.len(),
                    Some(variant_index),
                );
                PatKind::Variant {
                    adt_def,
                    substs,
                    variant_index,
                    subpatterns,
                }
            }
            ty::Adt(adt_def, _) => {
                let struct_var = adt_def.non_enum_variant();
                PatKind::Leaf {
                    subpatterns: adt_subpatterns(struct_var.fields.len(), None),
                }
            }
            ty::Tuple(fields) => {
                PatKind::Leaf {
                    subpatterns: adt_subpatterns(fields.len(), None),
                }
            }
            ty::Array(_, n) => {
                PatKind::Array {
                    prefix: (0..n.eval_usize(self.tcx, self.param_env))
                        .map(|i| adt_subpattern(i as usize, None))
                        .collect(),
                    slice: None,
                    suffix: Vec::new(),
                }
            }
            _ => {
                PatKind::Constant {
                    value: cv,
                }
            }
        };

        Pat {
            span,
            ty: cv.ty,
            kind: Box::new(kind),
        }
    }
}
