use rustc_hir as hir;
use rustc_hir::lang_items::EqTraitLangItem;
use rustc_index::vec::Idx;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_middle::mir::Field;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint;
use rustc_span::Span;
use rustc_trait_selection::traits::predicate_for_trait_def;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{self, ObligationCause, PredicateObligation};

use std::cell::Cell;

use super::{FieldPat, Pat, PatCtxt, PatKind};

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
    /// Converts an evaluated constant to a pattern (if possible).
    /// This means aggregate values (like structs and enums) are converted
    /// to a pattern that matches the value (as if you'd compared via structural equality).
    pub(super) fn const_to_pat(
        &self,
        cv: &'tcx ty::Const<'tcx>,
        id: hir::HirId,
        span: Span,
        mir_structural_match_violation: bool,
    ) -> Pat<'tcx> {
        debug!("const_to_pat: cv={:#?} id={:?}", cv, id);
        debug!("const_to_pat: cv.ty={:?} span={:?}", cv.ty, span);

        self.tcx.infer_ctxt().enter(|infcx| {
            let mut convert = ConstToPat::new(self, id, span, infcx);
            convert.to_pat(cv, mir_structural_match_violation)
        })
    }
}

struct ConstToPat<'a, 'tcx> {
    id: hir::HirId,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,

    // This tracks if we signal some hard error for a given const value, so that
    // we will not subsequently issue an irrelevant lint for the same const
    // value.
    saw_const_match_error: Cell<bool>,

    // inference context used for checking `T: Structural` bounds.
    infcx: InferCtxt<'a, 'tcx>,

    include_lint_checks: bool,
}

impl<'a, 'tcx> ConstToPat<'a, 'tcx> {
    fn new(
        pat_ctxt: &PatCtxt<'_, 'tcx>,
        id: hir::HirId,
        span: Span,
        infcx: InferCtxt<'a, 'tcx>,
    ) -> Self {
        ConstToPat {
            id,
            span,
            infcx,
            param_env: pat_ctxt.param_env,
            include_lint_checks: pat_ctxt.include_lint_checks,
            saw_const_match_error: Cell::new(false),
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn search_for_structural_match_violation(
        &self,
        ty: Ty<'tcx>,
    ) -> Option<traits::NonStructuralMatchTy<'tcx>> {
        traits::search_for_structural_match_violation(self.id, self.span, self.tcx(), ty)
    }

    fn type_marked_structural(&self, ty: Ty<'tcx>) -> bool {
        traits::type_marked_structural(self.id, self.span, &self.infcx, ty)
    }

    fn to_pat(
        &mut self,
        cv: &'tcx ty::Const<'tcx>,
        mir_structural_match_violation: bool,
    ) -> Pat<'tcx> {
        // This method is just a wrapper handling a validity check; the heavy lifting is
        // performed by the recursive `recur` method, which is not meant to be
        // invoked except by this method.
        //
        // once indirect_structural_match is a full fledged error, this
        // level of indirection can be eliminated

        let inlined_const_as_pat = self.recur(cv);

        if self.saw_const_match_error.get() {
            return inlined_const_as_pat;
        }

        // We eventually lower to a call to `PartialEq::eq` for this type, so ensure that this
        // method actually exists.
        if !self.ty_has_partial_eq_impl(cv.ty) {
            let msg = if cv.ty.is_trait() {
                "trait objects cannot be used in patterns".to_string()
            } else {
                format!("`{:?}` must implement `PartialEq` to be used in a pattern", cv.ty)
            };

            // Codegen will ICE if we continue compilation, so abort here.
            self.tcx().sess.span_fatal(self.span, &msg);
        }

        if !self.include_lint_checks {
            return inlined_const_as_pat;
        }

        // If we were able to successfully convert the const to some pat,
        // double-check that all types in the const implement `Structural`.

        let ty_violation = self.search_for_structural_match_violation(cv.ty);
        debug!(
            "search_for_structural_match_violation cv.ty: {:?} returned: {:?}",
            cv.ty, ty_violation,
        );

        if mir_structural_match_violation {
            let non_sm_ty =
                ty_violation.expect("MIR const-checker found novel structural match violation");
            let msg = match non_sm_ty {
                traits::NonStructuralMatchTy::Adt(adt_def) => {
                    let path = self.tcx().def_path_str(adt_def.did);
                    format!(
                        "to use a constant of type `{}` in a pattern, \
                             `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                        path, path,
                    )
                }
                traits::NonStructuralMatchTy::Dynamic => {
                    "trait objects cannot be used in patterns".to_string()
                }
                traits::NonStructuralMatchTy::Param => {
                    bug!("use of constant whose type is a parameter inside a pattern")
                }
            };

            self.tcx().struct_span_lint_hir(
                lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                self.id,
                self.span,
                |lint| lint.build(&msg).emit(),
            );
        } else if ty_violation.is_some() {
            debug!(
                "`search_for_structural_match_violation` found one, but `CustomEq` was \
                  not in the qualifs for that `const`"
            );
        }

        inlined_const_as_pat
    }

    // Recursive helper for `to_pat`; invoke that (instead of calling this directly).
    fn recur(&self, cv: &'tcx ty::Const<'tcx>) -> Pat<'tcx> {
        let id = self.id;
        let span = self.span;
        let tcx = self.tcx();
        let param_env = self.param_env;

        let field_pats = |vals: &[&'tcx ty::Const<'tcx>]| {
            vals.iter()
                .enumerate()
                .map(|(idx, val)| {
                    let field = Field::new(idx);
                    FieldPat { field, pattern: self.recur(val) }
                })
                .collect()
        };

        let kind = match cv.ty.kind {
            ty::Float(_) => {
                tcx.struct_span_lint_hir(
                    lint::builtin::ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
                    id,
                    span,
                    |lint| lint.build("floating-point types cannot be used in patterns").emit(),
                );
                PatKind::Constant { value: cv }
            }
            ty::Adt(adt_def, _) if adt_def.is_union() => {
                // Matching on union fields is unsafe, we can't hide it in constants
                self.saw_const_match_error.set(true);
                tcx.sess.span_err(span, "cannot use unions in constant patterns");
                PatKind::Wild
            }
            // keep old code until future-compat upgraded to errors.
            ty::Adt(adt_def, _) if !self.type_marked_structural(cv.ty) => {
                debug!("adt_def {:?} has !type_marked_structural for cv.ty: {:?}", adt_def, cv.ty);
                let path = tcx.def_path_str(adt_def.did);
                let msg = format!(
                    "to use a constant of type `{}` in a pattern, \
                     `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                    path, path,
                );
                self.saw_const_match_error.set(true);
                tcx.sess.span_err(span, &msg);
                PatKind::Wild
            }
            // keep old code until future-compat upgraded to errors.
            ty::Ref(_, adt_ty @ ty::TyS { kind: ty::Adt(_, _), .. }, _)
                if !self.type_marked_structural(adt_ty) =>
            {
                let adt_def =
                    if let ty::Adt(adt_def, _) = adt_ty.kind { adt_def } else { unreachable!() };

                debug!(
                    "adt_def {:?} has !type_marked_structural for adt_ty: {:?}",
                    adt_def, adt_ty
                );

                // HACK(estebank): Side-step ICE #53708, but anything other than erroring here
                // would be wrong. Returnging `PatKind::Wild` is not technically correct.
                let path = tcx.def_path_str(adt_def.did);
                let msg = format!(
                    "to use a constant of type `{}` in a pattern, \
                     `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                    path, path,
                );
                self.saw_const_match_error.set(true);
                tcx.sess.span_err(span, &msg);
                PatKind::Wild
            }
            ty::Adt(adt_def, substs) if adt_def.is_enum() => {
                let destructured = tcx.destructure_const(param_env.and(cv));
                PatKind::Variant {
                    adt_def,
                    substs,
                    variant_index: destructured.variant,
                    subpatterns: field_pats(destructured.fields),
                }
            }
            ty::Adt(_, _) => {
                let destructured = tcx.destructure_const(param_env.and(cv));
                PatKind::Leaf { subpatterns: field_pats(destructured.fields) }
            }
            ty::Tuple(_) => {
                let destructured = tcx.destructure_const(param_env.and(cv));
                PatKind::Leaf { subpatterns: field_pats(destructured.fields) }
            }
            ty::Array(..) => PatKind::Array {
                prefix: tcx
                    .destructure_const(param_env.and(cv))
                    .fields
                    .iter()
                    .map(|val| self.recur(val))
                    .collect(),
                slice: None,
                suffix: Vec::new(),
            },
            _ => PatKind::Constant { value: cv },
        };

        Pat { span, ty: cv.ty, kind: Box::new(kind) }
    }

    fn ty_has_partial_eq_impl(&self, ty: Ty<'tcx>) -> bool {
        let tcx = self.tcx();

        let is_partial_eq = |ty| {
            let partial_eq_trait_id = tcx.require_lang_item(EqTraitLangItem, Some(self.span));
            let obligation: PredicateObligation<'_> = predicate_for_trait_def(
                tcx,
                self.param_env,
                ObligationCause::misc(self.span, self.id),
                partial_eq_trait_id,
                0,
                ty,
                &[],
            );

            // FIXME: should this call a `predicate_must_hold` variant instead?
            self.infcx.predicate_may_hold(&obligation)
        };

        // Higher-ranked function pointers, such as `for<'r> fn(&'r i32)` are allowed in patterns
        // but do not satisfy `Self: PartialEq` due to shortcomings in the trait solver.
        // Check for bare function pointers first since it is cheap to do so.
        if let ty::FnPtr(_) = ty.kind {
            return true;
        }

        // In general, types that appear in patterns need to implement `PartialEq`.
        if is_partial_eq(ty) {
            return true;
        }

        // HACK: The check for bare function pointers will miss generic types that are instantiated
        // with a higher-ranked type (`for<'r> fn(&'r i32)`) as a parameter. To preserve backwards
        // compatibility in this case, we must continue to allow types such as `Option<fn(&i32)>`.
        //
        //
        // We accomplish this by replacing *all* late-bound lifetimes in the type with concrete
        // ones. This leverages the fact that function pointers with no late-bound lifetimes do
        // satisfy `PartialEq`. In other words, we transform `Option<for<'r> fn(&'r i32)>` to
        // `Option<fn(&'erased i32)>` and again check whether `PartialEq` is satisfied.
        // Obviously this is too permissive, but it is better than the old behavior, which
        // allowed *all* types to reach codegen and caused issues like #65466.
        let erased_ty = erase_all_late_bound_regions(tcx, ty);
        if is_partial_eq(erased_ty) {
            warn!("Non-function pointer only satisfied `PartialEq` after regions were erased");
            return true;
        }

        false
    }
}

/// Erase *all* late bound regions, ignoring their debruijn index.
///
/// This is a terrible hack. Do not use it elsewhere.
fn erase_all_late_bound_regions<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    use ty::fold::TypeFoldable;

    struct Eraser<'tcx> {
        tcx: TyCtxt<'tcx>,
    }

    impl<'tcx> ty::fold::TypeFolder<'tcx> for Eraser<'tcx> {
        fn tcx(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
            match r {
                ty::ReLateBound(_, _) => &ty::ReErased,
                r => r.super_fold_with(self),
            }
        }
    }

    ty.fold_with(&mut Eraser { tcx })
}
