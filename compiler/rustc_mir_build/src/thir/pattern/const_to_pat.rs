use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::Idx;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::Obligation;
use rustc_middle::mir;
use rustc_middle::thir::{FieldPat, Pat, PatKind};
use rustc_middle::ty::{self, Ty, TyCtxt, ValTree};
use rustc_session::lint;
use rustc_span::Span;
use rustc_target::abi::{FieldIdx, VariantIdx};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{self, ObligationCause};

use std::cell::Cell;

use super::PatCtxt;
use crate::errors::{
    FloatPattern, IndirectStructuralMatch, InvalidPattern, NontrivialStructuralMatch,
    PointerPattern, TypeNotStructural, UnionPattern, UnsizedPattern,
};

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
    /// Converts an evaluated constant to a pattern (if possible).
    /// This means aggregate values (like structs and enums) are converted
    /// to a pattern that matches the value (as if you'd compared via structural equality).
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn const_to_pat(
        &self,
        cv: mir::ConstantKind<'tcx>,
        id: hir::HirId,
        span: Span,
        check_body_for_struct_match_violation: Option<DefId>,
    ) -> Box<Pat<'tcx>> {
        let infcx = self.tcx.infer_ctxt().build();
        let mut convert = ConstToPat::new(self, id, span, infcx);
        convert.to_pat(cv, check_body_for_struct_match_violation)
    }
}

struct ConstToPat<'tcx> {
    id: hir::HirId,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,

    // This tracks if we emitted some hard error for a given const value, so that
    // we will not subsequently issue an irrelevant lint for the same const
    // value.
    saw_const_match_error: Cell<bool>,

    // This tracks if we emitted some diagnostic for a given const value, so that
    // we will not subsequently issue an irrelevant lint for the same const
    // value.
    saw_const_match_lint: Cell<bool>,

    // For backcompat we need to keep allowing non-structurally-eq types behind references.
    // See also all the `cant-hide-behind` tests.
    behind_reference: Cell<bool>,

    // inference context used for checking `T: Structural` bounds.
    infcx: InferCtxt<'tcx>,

    treat_byte_string_as_slice: bool,
}

/// This error type signals that we encountered a non-struct-eq situation.
/// We bubble this up in order to get back to the reference destructuring and make that emit
/// a const pattern instead of a deref pattern. This allows us to simply call `PartialEq::eq`
/// on such patterns (since that function takes a reference) and not have to jump through any
/// hoops to get a reference to the value.
#[derive(Debug)]
struct FallbackToConstRef;

impl<'tcx> ConstToPat<'tcx> {
    fn new(
        pat_ctxt: &PatCtxt<'_, 'tcx>,
        id: hir::HirId,
        span: Span,
        infcx: InferCtxt<'tcx>,
    ) -> Self {
        trace!(?pat_ctxt.typeck_results.hir_owner);
        ConstToPat {
            id,
            span,
            infcx,
            param_env: pat_ctxt.param_env,
            saw_const_match_error: Cell::new(false),
            saw_const_match_lint: Cell::new(false),
            behind_reference: Cell::new(false),
            treat_byte_string_as_slice: pat_ctxt
                .typeck_results
                .treat_byte_string_as_slice
                .contains(&id.local_id),
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn type_marked_structural(&self, ty: Ty<'tcx>) -> bool {
        ty.is_structural_eq_shallow(self.infcx.tcx)
    }

    fn to_pat(
        &mut self,
        cv: mir::ConstantKind<'tcx>,
        check_body_for_struct_match_violation: Option<DefId>,
    ) -> Box<Pat<'tcx>> {
        trace!(self.treat_byte_string_as_slice);
        // This method is just a wrapper handling a validity check; the heavy lifting is
        // performed by the recursive `recur` method, which is not meant to be
        // invoked except by this method.
        //
        // once indirect_structural_match is a full fledged error, this
        // level of indirection can be eliminated

        let mir_structural_match_violation = check_body_for_struct_match_violation.map(|def_id| {
            // `mir_const_qualif` must be called with the `DefId` of the item where the const is
            // defined, not where it is declared. The difference is significant for associated
            // constants.
            self.tcx().mir_const_qualif(def_id).custom_eq
        });
        debug!(?check_body_for_struct_match_violation, ?mir_structural_match_violation);

        let inlined_const_as_pat = match cv {
            mir::ConstantKind::Ty(c) => match c.kind() {
                ty::ConstKind::Param(_)
                | ty::ConstKind::Infer(_)
                | ty::ConstKind::Bound(_, _)
                | ty::ConstKind::Placeholder(_)
                | ty::ConstKind::Unevaluated(_)
                | ty::ConstKind::Error(_)
                | ty::ConstKind::Expr(_) => {
                    span_bug!(self.span, "unexpected const in `to_pat`: {:?}", c.kind())
                }
                ty::ConstKind::Value(valtree) => self
                    .recur(valtree, cv.ty(), mir_structural_match_violation.unwrap_or(false))
                    .unwrap_or_else(|_| {
                        Box::new(Pat {
                            span: self.span,
                            ty: cv.ty(),
                            kind: PatKind::Constant { value: cv },
                        })
                    }),
            },
            mir::ConstantKind::Unevaluated(_, _) => {
                span_bug!(self.span, "unevaluated const in `to_pat`: {cv:?}")
            }
            mir::ConstantKind::Val(_, _) => Box::new(Pat {
                span: self.span,
                ty: cv.ty(),
                kind: PatKind::Constant { value: cv },
            }),
        };

        if !self.saw_const_match_error.get() {
            // If we were able to successfully convert the const to some pat,
            // double-check that all types in the const implement `Structural`.

            let structural =
                traits::search_for_structural_match_violation(self.span, self.tcx(), cv.ty());
            debug!(
                "search_for_structural_match_violation cv.ty: {:?} returned: {:?}",
                cv.ty(),
                structural
            );

            // This can occur because const qualification treats all associated constants as
            // opaque, whereas `search_for_structural_match_violation` tries to monomorphize them
            // before it runs.
            //
            // FIXME(#73448): Find a way to bring const qualification into parity with
            // `search_for_structural_match_violation`.
            if structural.is_none() && mir_structural_match_violation.unwrap_or(false) {
                warn!("MIR const-checker found novel structural match violation. See #73448.");
                return inlined_const_as_pat;
            }

            if let Some(non_sm_ty) = structural {
                if !self.type_may_have_partial_eq_impl(cv.ty()) {
                    if let ty::Adt(def, ..) = non_sm_ty.kind() {
                        if def.is_union() {
                            let err = UnionPattern { span: self.span };
                            self.tcx().sess.emit_err(err);
                        } else {
                            // fatal avoids ICE from resolution of nonexistent method (rare case).
                            self.tcx()
                                .sess
                                .emit_fatal(TypeNotStructural { span: self.span, non_sm_ty });
                        }
                    } else {
                        let err = InvalidPattern { span: self.span, non_sm_ty };
                        self.tcx().sess.emit_err(err);
                        return Box::new(Pat { span: self.span, ty: cv.ty(), kind: PatKind::Wild });
                    }
                } else if !self.saw_const_match_lint.get() {
                    if let Some(mir_structural_match_violation) = mir_structural_match_violation {
                        match non_sm_ty.kind() {
                            ty::RawPtr(pointee)
                                if pointee.ty.is_sized(self.tcx(), self.param_env) => {}
                            ty::FnPtr(..) | ty::RawPtr(..) => {
                                self.tcx().emit_spanned_lint(
                                    lint::builtin::POINTER_STRUCTURAL_MATCH,
                                    self.id,
                                    self.span,
                                    PointerPattern,
                                );
                            }
                            ty::Adt(..) if mir_structural_match_violation => {
                                self.tcx().emit_spanned_lint(
                                    lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                                    self.id,
                                    self.span,
                                    IndirectStructuralMatch { non_sm_ty },
                                );
                            }
                            _ => {
                                debug!(
                                    "`search_for_structural_match_violation` found one, but `CustomEq` was \
                                  not in the qualifs for that `const`"
                                );
                            }
                        }
                    }
                }
            } else if !self.saw_const_match_lint.get() {
                match cv.ty().kind() {
                    ty::RawPtr(pointee) if pointee.ty.is_sized(self.tcx(), self.param_env) => {}
                    ty::FnPtr(..) | ty::RawPtr(..) => {
                        self.tcx().emit_spanned_lint(
                            lint::builtin::POINTER_STRUCTURAL_MATCH,
                            self.id,
                            self.span,
                            PointerPattern,
                        );
                    }
                    _ => {}
                }
            }
        }

        inlined_const_as_pat
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn type_may_have_partial_eq_impl(&self, ty: Ty<'tcx>) -> bool {
        // double-check there even *is* a semantic `PartialEq` to dispatch to.
        //
        // (If there isn't, then we can safely issue a hard
        // error, because that's never worked, due to compiler
        // using `PartialEq::eq` in this scenario in the past.)
        let partial_eq_trait_id =
            self.tcx().require_lang_item(hir::LangItem::PartialEq, Some(self.span));
        let partial_eq_obligation = Obligation::new(
            self.tcx(),
            ObligationCause::dummy(),
            self.param_env,
            ty::TraitRef::new(self.tcx(), partial_eq_trait_id, [ty, ty]),
        );

        // FIXME: should this call a `predicate_must_hold` variant instead?
        self.infcx.predicate_may_hold(&partial_eq_obligation)
    }

    fn field_pats(
        &self,
        vals: impl Iterator<Item = (ValTree<'tcx>, Ty<'tcx>)>,
    ) -> Result<Vec<FieldPat<'tcx>>, FallbackToConstRef> {
        vals.enumerate()
            .map(|(idx, (val, ty))| {
                let field = FieldIdx::new(idx);
                // Patterns can only use monomorphic types.
                let ty = self.tcx().normalize_erasing_regions(self.param_env, ty);
                Ok(FieldPat { field, pattern: self.recur(val, ty, false)? })
            })
            .collect()
    }

    // Recursive helper for `to_pat`; invoke that (instead of calling this directly).
    #[instrument(skip(self), level = "debug")]
    fn recur(
        &self,
        cv: ValTree<'tcx>,
        ty: Ty<'tcx>,
        mir_structural_match_violation: bool,
    ) -> Result<Box<Pat<'tcx>>, FallbackToConstRef> {
        let id = self.id;
        let span = self.span;
        let tcx = self.tcx();
        let param_env = self.param_env;

        let kind = match ty.kind() {
            ty::Float(_) => {
                self.saw_const_match_lint.set(true);
                tcx.emit_spanned_lint(
                    lint::builtin::ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
                    id,
                    span,
                    FloatPattern,
                );
                return Err(FallbackToConstRef);
            }
            // If the type is not structurally comparable, just emit the constant directly,
            // causing the pattern match code to treat it opaquely.
            // FIXME: This code doesn't emit errors itself, the caller emits the errors.
            // So instead of specific errors, you just get blanket errors about the whole
            // const type. See
            // https://github.com/rust-lang/rust/pull/70743#discussion_r404701963 for
            // details.
            // Backwards compatibility hack because we can't cause hard errors on these
            // types, so we compare them via `PartialEq::eq` at runtime.
            ty::Adt(..) if !self.type_marked_structural(ty) && self.behind_reference.get() => {
                if !self.saw_const_match_error.get() && !self.saw_const_match_lint.get() {
                    self.saw_const_match_lint.set(true);
                    tcx.emit_spanned_lint(
                        lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                        id,
                        span,
                        IndirectStructuralMatch { non_sm_ty: ty },
                    );
                }
                // Since we are behind a reference, we can just bubble the error up so we get a
                // constant at reference type, making it easy to let the fallback call
                // `PartialEq::eq` on it.
                return Err(FallbackToConstRef);
            }
            ty::Adt(adt_def, _) if !self.type_marked_structural(ty) => {
                debug!("adt_def {:?} has !type_marked_structural for cv.ty: {:?}", adt_def, ty,);
                self.saw_const_match_error.set(true);
                let err = TypeNotStructural { span, non_sm_ty: ty };
                tcx.sess.emit_err(err);
                PatKind::Wild
            }
            ty::Adt(adt_def, substs) if adt_def.is_enum() => {
                let (&variant_index, fields) = cv.unwrap_branch().split_first().unwrap();
                let variant_index =
                    VariantIdx::from_u32(variant_index.unwrap_leaf().try_to_u32().ok().unwrap());
                PatKind::Variant {
                    adt_def: *adt_def,
                    substs,
                    variant_index,
                    subpatterns: self.field_pats(
                        fields.iter().copied().zip(
                            adt_def.variants()[variant_index]
                                .fields
                                .iter()
                                .map(|field| field.ty(self.tcx(), substs)),
                        ),
                    )?,
                }
            }
            ty::Tuple(fields) => PatKind::Leaf {
                subpatterns: self
                    .field_pats(cv.unwrap_branch().iter().copied().zip(fields.iter()))?,
            },
            ty::Adt(def, substs) => PatKind::Leaf {
                subpatterns: self.field_pats(cv.unwrap_branch().iter().copied().zip(
                    def.non_enum_variant().fields.iter().map(|field| field.ty(self.tcx(), substs)),
                ))?,
            },
            ty::Array(elem_ty, _) => PatKind::Array {
                prefix: cv
                    .unwrap_branch()
                    .iter()
                    .map(|val| self.recur(*val, *elem_ty, false))
                    .collect::<Result<_, _>>()?,
                slice: None,
                suffix: Box::new([]),
            },
            ty::Ref(_, pointee_ty, ..) => match *pointee_ty.kind() {
                // `&str` is represented as a valtree, let's keep using this
                // optimization for now.
                ty::Str => PatKind::Constant { value: mir::ConstantKind::Ty(tcx.mk_const(cv, ty)) },
                // `b"foo"` produces a `&[u8; 3]`, but you can't use constants of array type when
                // matching against references, you can only use byte string literals.
                // The typechecker has a special case for byte string literals, by treating them
                // as slices. This means we turn `&[T; N]` constants into slice patterns, which
                // has no negative effects on pattern matching, even if we're actually matching on
                // arrays.
                ty::Array(elem_ty, _) if !self.treat_byte_string_as_slice => {
                    let old = self.behind_reference.replace(true);
                    // References have the same valtree representation as their pointee.
                    let array = cv;
                    let val = PatKind::Deref {
                        subpattern: Box::new(Pat {
                            kind: PatKind::Array {
                                prefix: array.unwrap_branch()
                                    .iter()
                                    .map(|val| self.recur(*val, elem_ty, false))
                                    .collect::<Result<_, _>>()?,
                                slice: None,
                                suffix: Box::new([]),
                            },
                            span,
                            ty: tcx.mk_slice(elem_ty),
                        }),
                    };
                    self.behind_reference.set(old);
                    val
                }
                ty::Array(elem_ty, _) |
                // Cannot merge this with the catch all branch below, because the `const_deref`
                // changes the type from slice to array, we need to keep the original type in the
                // pattern.
                ty::Slice(elem_ty) => {
                    let old = self.behind_reference.replace(true);
                    // References have the same valtree representation as their pointee.
                    let array = cv;
                    let val = PatKind::Deref {
                        subpattern: Box::new(Pat {
                            kind: PatKind::Slice {
                                prefix: array.unwrap_branch()
                                    .iter()
                                    .map(|val| self.recur(*val, elem_ty, false))
                                    .collect::<Result<_, _>>()?,
                                slice: None,
                                suffix: Box::new([]),
                            },
                            span,
                            ty: tcx.mk_slice(elem_ty),
                        }),
                    };
                    self.behind_reference.set(old);
                    val
                }
                // Backwards compatibility hack: support references to non-structural types,
                // but hard error if we aren't behind a double reference. We could just use
                // the fallback code path below, but that would allow *more* of this fishy
                // code to compile, as then it only goes through the future incompat lint
                // instead of a hard error.
                ty::Adt(_, _) if !self.type_marked_structural(*pointee_ty) => {
                    if self.behind_reference.get() {
                        if !self.saw_const_match_error.get()
                            && !self.saw_const_match_lint.get()
                        {
                           self.saw_const_match_lint.set(true);
                           tcx.emit_spanned_lint(
                                lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                                self.id,
                                span,
                                IndirectStructuralMatch { non_sm_ty: *pointee_ty },
                            );
                        }
                        return Err(FallbackToConstRef);
                    } else {
                        if !self.saw_const_match_error.get() {
                            self.saw_const_match_error.set(true);
                            let err = TypeNotStructural { span, non_sm_ty: *pointee_ty };
                            tcx.sess.emit_err(err);
                        }
                        PatKind::Wild
                    }
                }
                // All other references are converted into deref patterns and then recursively
                // convert the dereferenced constant to a pattern that is the sub-pattern of the
                // deref pattern.
                _ => {
                    if !pointee_ty.is_sized(tcx, param_env) {
                        let err = UnsizedPattern { span, non_sm_ty: *pointee_ty };
                        tcx.sess.emit_err(err);

                        // FIXME: introduce PatKind::Error to silence follow up diagnostics due to unreachable patterns.
                        PatKind::Wild
                    } else {
                        let old = self.behind_reference.replace(true);
                        // References have the same valtree representation as their pointee.
                        let subpattern = self.recur(cv, *pointee_ty, false)?;
                        self.behind_reference.set(old);
                        PatKind::Deref { subpattern }
                    }
                }
            },
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::FnDef(..) => {
                PatKind::Constant { value: mir::ConstantKind::Ty(tcx.mk_const(cv, ty)) }
            }
            ty::FnPtr(..) | ty::RawPtr(..) => unreachable!(),
            _ => {
                self.saw_const_match_error.set(true);
                let err = InvalidPattern { span, non_sm_ty: ty };
                tcx.sess.emit_err(err);
                PatKind::Wild
            }
        };

        if !self.saw_const_match_error.get()
            && !self.saw_const_match_lint.get()
            && mir_structural_match_violation
            // FIXME(#73448): Find a way to bring const qualification into parity with
            // `search_for_structural_match_violation` and then remove this condition.

            // Obtain the actual type that isn't annotated. If we just looked at `cv.ty` we
            // could get `Option<NonStructEq>`, even though `Option` is annotated with derive.
            && let Some(non_sm_ty) = traits::search_for_structural_match_violation(span, tcx, ty)
        {
            self.saw_const_match_lint.set(true);
            tcx.emit_spanned_lint(
                lint::builtin::NONTRIVIAL_STRUCTURAL_MATCH,
                id,
                span,
                NontrivialStructuralMatch {non_sm_ty}
            );
        }

        Ok(Box::new(Pat { span, ty, kind }))
    }
}
