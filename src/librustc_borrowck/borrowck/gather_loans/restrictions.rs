//! Computes the restrictions that result from a borrow.

use crate::borrowck::*;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::ty;
use syntax_pos::Span;
use log::debug;

use crate::borrowck::ToInteriorKind;

use std::rc::Rc;

#[derive(Debug)]
pub enum RestrictionResult<'tcx> {
    Safe,
    SafeIf(Rc<LoanPath<'tcx>>, Vec<Rc<LoanPath<'tcx>>>)
}

pub fn compute_restrictions<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                      span: Span,
                                      cause: euv::LoanCause,
                                      cmt: &mc::cmt_<'tcx>,
                                      loan_region: ty::Region<'tcx>)
                                      -> RestrictionResult<'tcx> {
    let ctxt = RestrictionsContext {
        bccx,
        span,
        cause,
        loan_region,
    };

    ctxt.restrict(cmt)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct RestrictionsContext<'a, 'tcx> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    span: Span,
    loan_region: ty::Region<'tcx>,
    cause: euv::LoanCause,
}

impl<'a, 'tcx> RestrictionsContext<'a, 'tcx> {
    fn restrict(&self,
                cmt: &mc::cmt_<'tcx>) -> RestrictionResult<'tcx> {
        debug!("restrict(cmt={:?})", cmt);

        let new_lp = |v: LoanPathKind<'tcx>| Rc::new(LoanPath::new(v, cmt.ty));

        match cmt.cat.clone() {
            Categorization::Rvalue(..) => {
                // Effectively, rvalues are stored into a
                // non-aliasable temporary on the stack. Since they
                // are inherently non-aliasable, they can only be
                // accessed later through the borrow itself and hence
                // must inherently comply with its terms.
                RestrictionResult::Safe
            }

            Categorization::ThreadLocal(..) => {
                // Thread-locals are statics that have a scope, with
                // no underlying structure to provide restrictions.
                RestrictionResult::Safe
            }

            Categorization::Local(local_id) => {
                // R-Variable, locally declared
                let lp = new_lp(LpVar(local_id));
                RestrictionResult::SafeIf(lp.clone(), vec![lp])
            }

            Categorization::Upvar(mc::Upvar { id, .. }) => {
                // R-Variable, captured into closure
                let lp = new_lp(LpUpvar(id));
                RestrictionResult::SafeIf(lp.clone(), vec![lp])
            }

            Categorization::Downcast(cmt_base, _) => {
                // When we borrow the interior of an enum, we have to
                // ensure the enum itself is not mutated, because that
                // could cause the type of the memory to change.
                self.restrict(&cmt_base)
            }

            Categorization::Interior(cmt_base, interior) => {
                // R-Field
                //
                // Overwriting the base would not change the type of
                // the memory, so no additional restrictions are
                // needed.
                let opt_variant_id = match cmt_base.cat {
                    Categorization::Downcast(_, variant_id) => Some(variant_id),
                    _ => None
                };
                let interior = interior.cleaned();
                let base_ty = cmt_base.ty;
                let result = self.restrict(&cmt_base);
                // Borrowing one union field automatically borrows all its fields.
                match base_ty.sty {
                    ty::Adt(adt_def, _) if adt_def.is_union() => match result {
                        RestrictionResult::Safe => RestrictionResult::Safe,
                        RestrictionResult::SafeIf(base_lp, mut base_vec) => {
                            for (i, field) in adt_def.non_enum_variant().fields.iter().enumerate() {
                                let field = InteriorKind::InteriorField(
                                    mc::FieldIndex(i, field.ident.name)
                                );
                                let field_ty = if field == interior {
                                    cmt.ty
                                } else {
                                    self.bccx.tcx.types.err // Doesn't matter
                                };
                                let sibling_lp_kind = LpExtend(base_lp.clone(), cmt.mutbl,
                                                               LpInterior(opt_variant_id, field));
                                let sibling_lp = Rc::new(LoanPath::new(sibling_lp_kind, field_ty));
                                base_vec.push(sibling_lp);
                            }

                            let lp = new_lp(LpExtend(base_lp, cmt.mutbl,
                                                     LpInterior(opt_variant_id, interior)));
                            RestrictionResult::SafeIf(lp, base_vec)
                        }
                    },
                    _ => self.extend(result, &cmt, LpInterior(opt_variant_id, interior))
                }
            }

            Categorization::StaticItem => {
                RestrictionResult::Safe
            }

            Categorization::Deref(cmt_base, pk) => {
                match pk {
                    mc::Unique => {
                        // R-Deref-Send-Pointer
                        //
                        // When we borrow the interior of a box, we
                        // cannot permit the base to be mutated, because that
                        // would cause the unique pointer to be freed.
                        //
                        // Eventually we should make these non-special and
                        // just rely on Deref<T> implementation.
                        let result = self.restrict(&cmt_base);
                        self.extend(result, &cmt, LpDeref(pk))
                    }
                    mc::BorrowedPtr(bk, lt) => {
                        // R-Deref-[Mut-]Borrowed
                        if !self.bccx.is_subregion_of(self.loan_region, lt) {
                            self.bccx.report(
                                BckError {
                                    span: self.span,
                                    cause: BorrowViolation(self.cause),
                                    cmt: &cmt_base,
                                    code: err_borrowed_pointer_too_short(
                                        self.loan_region, lt)});
                            return RestrictionResult::Safe;
                        }

                        match bk {
                            ty::ImmBorrow => RestrictionResult::Safe,
                            ty::MutBorrow | ty::UniqueImmBorrow => {
                                // R-Deref-Mut-Borrowed
                                //
                                // The referent can be aliased after the
                                // references lifetime ends (by a newly-unfrozen
                                // borrow).
                                let result = self.restrict(&cmt_base);
                                self.extend(result, &cmt, LpDeref(pk))
                            }
                        }
                    }
                    // Borrowck is not relevant for raw pointers
                    mc::UnsafePtr(..) => RestrictionResult::Safe
                }
            }
        }
    }

    fn extend(&self,
              result: RestrictionResult<'tcx>,
              cmt: &mc::cmt_<'tcx>,
              elem: LoanPathElem<'tcx>) -> RestrictionResult<'tcx> {
        match result {
            RestrictionResult::Safe => RestrictionResult::Safe,
            RestrictionResult::SafeIf(base_lp, mut base_vec) => {
                let v = LpExtend(base_lp, cmt.mutbl, elem);
                let lp = Rc::new(LoanPath::new(v, cmt.ty));
                base_vec.push(lp.clone());
                RestrictionResult::SafeIf(lp, base_vec)
            }
        }
    }
}
