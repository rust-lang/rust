use rustc::infer::InferCtxt;
use rustc::lint;
use rustc::mir::Field;
use rustc::traits::predicate_for_trait_def;
use rustc::traits::{self, ObligationCause, PredicateObligation};
use rustc::ty::{self, Ty, TyCtxt};
use rustc_hir as hir;

use rustc_index::vec::Idx;

use rustc_span::Span;

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
    ) -> Pat<'tcx> {
        debug!("const_to_pat: cv={:#?} id={:?}", cv, id);
        debug!("const_to_pat: cv.ty={:?} span={:?}", cv.ty, span);

        self.tcx.infer_ctxt().enter(|infcx| {
            let mut convert = ConstToPat::new(self, id, span, infcx);
            convert.to_pat(cv)
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

    fn to_pat(&mut self, cv: &'tcx ty::Const<'tcx>) -> Pat<'tcx> {
        // This method is just a wrapper handling a validity check; the heavy lifting is
        // performed by the recursive `recur` method, which is not meant to be
        // invoked except by this method.
        //
        // once indirect_structural_match is a full fledged error, this
        // level of indirection can be eliminated

        let inlined_const_as_pat = self.recur(cv);

        if self.include_lint_checks && !self.saw_const_match_error.get() {
            // If we were able to successfully convert the const to some pat,
            // double-check that all types in the const implement `Structural`.

            let structural = self.search_for_structural_match_violation(cv.ty);
            debug!(
                "search_for_structural_match_violation cv.ty: {:?} returned: {:?}",
                cv.ty, structural
            );
            if let Some(non_sm_ty) = structural {
                let adt_def = match non_sm_ty {
                    traits::NonStructuralMatchTy::Adt(adt_def) => adt_def,
                    traits::NonStructuralMatchTy::Param => {
                        bug!("use of constant whose type is a parameter inside a pattern")
                    }
                };
                let path = self.tcx().def_path_str(adt_def.did);

                let make_msg = || -> String {
                    format!(
                        "to use a constant of type `{}` in a pattern, \
                         `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                        path, path,
                    )
                };

                // double-check there even *is* a semantic `PartialEq` to dispatch to.
                //
                // (If there isn't, then we can safely issue a hard
                // error, because that's never worked, due to compiler
                // using `PartialEq::eq` in this scenario in the past.)
                //
                // Note: To fix rust-lang/rust#65466, one could lift this check
                // *before* any structural-match checking, and unconditionally error
                // if `PartialEq` is not implemented. However, that breaks stable
                // code at the moment, because types like `for <'a> fn(&'a ())` do
                // not *yet* implement `PartialEq`. So for now we leave this here.
                let ty_is_partial_eq: bool = {
                    let partial_eq_trait_id = self.tcx().lang_items().eq_trait().unwrap();
                    let obligation: PredicateObligation<'_> = predicate_for_trait_def(
                        self.tcx(),
                        self.param_env,
                        ObligationCause::misc(self.span, self.id),
                        partial_eq_trait_id,
                        0,
                        cv.ty,
                        &[],
                    );
                    // FIXME: should this call a `predicate_must_hold` variant instead?
                    self.infcx.predicate_may_hold(&obligation)
                };

                if !ty_is_partial_eq {
                    // span_fatal avoids ICE from resolution of non-existent method (rare case).
                    self.tcx().sess.span_fatal(self.span, &make_msg());
                } else {
                    self.tcx().struct_span_lint_hir(
                        lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                        self.id,
                        self.span,
                        |lint| lint.build(&make_msg()).emit(),
                    );
                }
            }
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
                    ::rustc::lint::builtin::ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
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
}
