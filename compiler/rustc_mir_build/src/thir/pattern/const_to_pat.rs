use rustc_hir as hir;
use rustc_index::vec::Idx;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_middle::mir::Field;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
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

        let pat = self.tcx.infer_ctxt().enter(|infcx| {
            let mut convert = ConstToPat::new(self, id, span, infcx);
            convert.to_pat(cv, mir_structural_match_violation)
        });

        debug!("const_to_pat: pat={:?}", pat);
        pat
    }
}

struct ConstToPat<'a, 'tcx> {
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
    infcx: InferCtxt<'a, 'tcx>,

    include_lint_checks: bool,
}

mod fallback_to_const_ref {
    #[derive(Debug)]
    /// This error type signals that we encountered a non-struct-eq situation behind a reference.
    /// We bubble this up in order to get back to the reference destructuring and make that emit
    /// a const pattern instead of a deref pattern. This allows us to simply call `PartialEq::eq`
    /// on such patterns (since that function takes a reference) and not have to jump through any
    /// hoops to get a reference to the value.
    pub(super) struct FallbackToConstRef(());

    pub(super) fn fallback_to_const_ref<'a, 'tcx>(
        c2p: &super::ConstToPat<'a, 'tcx>,
    ) -> FallbackToConstRef {
        assert!(c2p.behind_reference.get());
        FallbackToConstRef(())
    }
}
use fallback_to_const_ref::{fallback_to_const_ref, FallbackToConstRef};

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
            saw_const_match_lint: Cell::new(false),
            behind_reference: Cell::new(false),
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn adt_derive_msg(&self, adt_def: &AdtDef) -> String {
        let path = self.tcx().def_path_str(adt_def.did);
        format!(
            "to use a constant of type `{}` in a pattern, \
            `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
            path, path,
        )
    }

    fn search_for_structural_match_violation(&self, ty: Ty<'tcx>) -> Option<String> {
        traits::search_for_structural_match_violation(self.id, self.span, self.tcx(), ty).map(
            |non_sm_ty| {
                with_no_trimmed_paths(|| match non_sm_ty {
                    traits::NonStructuralMatchTy::Adt(adt) => self.adt_derive_msg(adt),
                    traits::NonStructuralMatchTy::Dynamic => {
                        "trait objects cannot be used in patterns".to_string()
                    }
                    traits::NonStructuralMatchTy::Opaque => {
                        "opaque types cannot be used in patterns".to_string()
                    }
                    traits::NonStructuralMatchTy::Generator => {
                        "generators cannot be used in patterns".to_string()
                    }
                    traits::NonStructuralMatchTy::Closure => {
                        "closures cannot be used in patterns".to_string()
                    }
                    traits::NonStructuralMatchTy::Param => {
                        bug!("use of a constant whose type is a parameter inside a pattern")
                    }
                    traits::NonStructuralMatchTy::Projection => {
                        bug!("use of a constant whose type is a projection inside a pattern")
                    }
                    traits::NonStructuralMatchTy::Foreign => {
                        bug!("use of a value of a foreign type inside a pattern")
                    }
                })
            },
        )
    }

    fn type_marked_structural(&self, ty: Ty<'tcx>) -> bool {
        ty.is_structural_eq_shallow(self.infcx.tcx)
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

        let inlined_const_as_pat = self.recur(cv, mir_structural_match_violation).unwrap();

        if self.include_lint_checks && !self.saw_const_match_error.get() {
            // If we were able to successfully convert the const to some pat,
            // double-check that all types in the const implement `Structural`.

            let structural = self.search_for_structural_match_violation(cv.ty);
            debug!(
                "search_for_structural_match_violation cv.ty: {:?} returned: {:?}",
                cv.ty, structural
            );

            // This can occur because const qualification treats all associated constants as
            // opaque, whereas `search_for_structural_match_violation` tries to monomorphize them
            // before it runs.
            //
            // FIXME(#73448): Find a way to bring const qualification into parity with
            // `search_for_structural_match_violation`.
            if structural.is_none() && mir_structural_match_violation {
                warn!("MIR const-checker found novel structural match violation. See #73448.");
                return inlined_const_as_pat;
            }

            if let Some(msg) = structural {
                if !self.type_may_have_partial_eq_impl(cv.ty) {
                    // span_fatal avoids ICE from resolution of non-existent method (rare case).
                    self.tcx().sess.span_fatal(self.span, &msg);
                } else if mir_structural_match_violation && !self.saw_const_match_lint.get() {
                    self.tcx().struct_span_lint_hir(
                        lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                        self.id,
                        self.span,
                        |lint| lint.build(&msg).emit(),
                    );
                } else {
                    debug!(
                        "`search_for_structural_match_violation` found one, but `CustomEq` was \
                          not in the qualifs for that `const`"
                    );
                }
            }
        }

        inlined_const_as_pat
    }

    fn type_may_have_partial_eq_impl(&self, ty: Ty<'tcx>) -> bool {
        // double-check there even *is* a semantic `PartialEq` to dispatch to.
        //
        // (If there isn't, then we can safely issue a hard
        // error, because that's never worked, due to compiler
        // using `PartialEq::eq` in this scenario in the past.)
        let partial_eq_trait_id =
            self.tcx().require_lang_item(hir::LangItem::PartialEq, Some(self.span));
        let obligation: PredicateObligation<'_> = predicate_for_trait_def(
            self.tcx(),
            self.param_env,
            ObligationCause::misc(self.span, self.id),
            partial_eq_trait_id,
            0,
            ty,
            &[],
        );
        // FIXME: should this call a `predicate_must_hold` variant instead?

        let has_impl = self.infcx.predicate_may_hold(&obligation);

        // Note: To fix rust-lang/rust#65466, we could just remove this type
        // walk hack for function pointers, and unconditionally error
        // if `PartialEq` is not implemented. However, that breaks stable
        // code at the moment, because types like `for <'a> fn(&'a ())` do
        // not *yet* implement `PartialEq`. So for now we leave this here.
        has_impl
            || ty.walk().any(|t| match t.unpack() {
                ty::subst::GenericArgKind::Lifetime(_) => false,
                ty::subst::GenericArgKind::Type(t) => t.is_fn_ptr(),
                ty::subst::GenericArgKind::Const(_) => false,
            })
    }

    // Recursive helper for `to_pat`; invoke that (instead of calling this directly).
    fn recur(
        &self,
        cv: &'tcx ty::Const<'tcx>,
        mir_structural_match_violation: bool,
    ) -> Result<Pat<'tcx>, FallbackToConstRef> {
        let id = self.id;
        let span = self.span;
        let tcx = self.tcx();
        let param_env = self.param_env;

        let field_pats = |vals: &[&'tcx ty::Const<'tcx>]| -> Result<_, _> {
            vals.iter()
                .enumerate()
                .map(|(idx, val)| {
                    let field = Field::new(idx);
                    Ok(FieldPat { field, pattern: self.recur(val, false)? })
                })
                .collect()
        };

        let kind = match cv.ty.kind() {
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
                let msg = "cannot use unions in constant patterns";
                if self.include_lint_checks {
                    tcx.sess.span_err(span, msg);
                } else {
                    tcx.sess.delay_span_bug(span, msg)
                }
                PatKind::Wild
            }
            ty::Adt(..)
                if !self.type_may_have_partial_eq_impl(cv.ty)
                    // FIXME(#73448): Find a way to bring const qualification into parity with
                    // `search_for_structural_match_violation` and then remove this condition.
                    && self.search_for_structural_match_violation(cv.ty).is_some() =>
            {
                // Obtain the actual type that isn't annotated. If we just looked at `cv.ty` we
                // could get `Option<NonStructEq>`, even though `Option` is annotated with derive.
                let msg = self.search_for_structural_match_violation(cv.ty).unwrap();
                self.saw_const_match_error.set(true);
                if self.include_lint_checks {
                    tcx.sess.span_err(self.span, &msg);
                } else {
                    tcx.sess.delay_span_bug(self.span, &msg)
                }
                PatKind::Wild
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
            ty::Adt(..) if !self.type_marked_structural(cv.ty) && self.behind_reference.get() => {
                if self.include_lint_checks
                    && !self.saw_const_match_error.get()
                    && !self.saw_const_match_lint.get()
                {
                    self.saw_const_match_lint.set(true);
                    let msg = format!(
                        "to use a constant of type `{}` in a pattern, \
                        `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                        cv.ty, cv.ty,
                    );
                    tcx.struct_span_lint_hir(
                        lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                        id,
                        span,
                        |lint| lint.build(&msg).emit(),
                    );
                }
                // Since we are behind a reference, we can just bubble the error up so we get a
                // constant at reference type, making it easy to let the fallback call
                // `PartialEq::eq` on it.
                return Err(fallback_to_const_ref(self));
            }
            ty::Adt(adt_def, _) if !self.type_marked_structural(cv.ty) => {
                debug!("adt_def {:?} has !type_marked_structural for cv.ty: {:?}", adt_def, cv.ty);
                let path = tcx.def_path_str(adt_def.did);
                let msg = format!(
                    "to use a constant of type `{}` in a pattern, \
                     `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                    path, path,
                );
                self.saw_const_match_error.set(true);
                if self.include_lint_checks {
                    tcx.sess.span_err(span, &msg);
                } else {
                    tcx.sess.delay_span_bug(span, &msg)
                }
                PatKind::Wild
            }
            ty::Adt(adt_def, substs) if adt_def.is_enum() => {
                let destructured = tcx.destructure_const(param_env.and(cv));
                PatKind::Variant {
                    adt_def,
                    substs,
                    variant_index: destructured
                        .variant
                        .expect("destructed const of adt without variant id"),
                    subpatterns: field_pats(destructured.fields)?,
                }
            }
            ty::Tuple(_) | ty::Adt(_, _) => {
                let destructured = tcx.destructure_const(param_env.and(cv));
                PatKind::Leaf { subpatterns: field_pats(destructured.fields)? }
            }
            ty::Array(..) => PatKind::Array {
                prefix: tcx
                    .destructure_const(param_env.and(cv))
                    .fields
                    .iter()
                    .map(|val| self.recur(val, false))
                    .collect::<Result<_, _>>()?,
                slice: None,
                suffix: Vec::new(),
            },
            ty::Ref(_, pointee_ty, ..) => match *pointee_ty.kind() {
                // These are not allowed and will error elsewhere anyway.
                ty::Dynamic(..) => {
                    self.saw_const_match_error.set(true);
                    let msg = format!("`{}` cannot be used in patterns", cv.ty);
                    if self.include_lint_checks {
                        tcx.sess.span_err(span, &msg);
                    } else {
                        tcx.sess.delay_span_bug(span, &msg)
                    }
                    PatKind::Wild
                }
                // `&str` and `&[u8]` are represented as `ConstValue::Slice`, let's keep using this
                // optimization for now.
                ty::Str => PatKind::Constant { value: cv },
                // `b"foo"` produces a `&[u8; 3]`, but you can't use constants of array type when
                // matching against references, you can only use byte string literals.
                // The typechecker has a special case for byte string literals, by treating them
                // as slices. This means we turn `&[T; N]` constants into slice patterns, which
                // has no negative effects on pattern matching, even if we're actually matching on
                // arrays.
                ty::Array(..) |
                // Cannot merge this with the catch all branch below, because the `const_deref`
                // changes the type from slice to array, we need to keep the original type in the
                // pattern.
                ty::Slice(..) => {
                    let old = self.behind_reference.replace(true);
                    let array = tcx.deref_const(self.param_env.and(cv));
                    let val = PatKind::Deref {
                        subpattern: Pat {
                            kind: Box::new(PatKind::Slice {
                                prefix: tcx
                                    .destructure_const(param_env.and(array))
                                    .fields
                                    .iter()
                                    .map(|val| self.recur(val, false))
                                    .collect::<Result<_, _>>()?,
                                slice: None,
                                suffix: vec![],
                            }),
                            span,
                            ty: pointee_ty,
                        },
                    };
                    self.behind_reference.set(old);
                    val
                }
                // Backwards compatibility hack: support references to non-structural types.
                // We'll lower
                // this pattern to a `PartialEq::eq` comparison and `PartialEq::eq` takes a
                // reference. This makes the rest of the matching logic simpler as it doesn't have
                // to figure out how to get a reference again.
                ty::Adt(adt_def, _) if !self.type_marked_structural(pointee_ty) => {
                    if self.behind_reference.get() {
                        if self.include_lint_checks
                            && !self.saw_const_match_error.get()
                            && !self.saw_const_match_lint.get()
                        {
                            self.saw_const_match_lint.set(true);
                            let msg = self.adt_derive_msg(adt_def);
                            self.tcx().struct_span_lint_hir(
                                lint::builtin::INDIRECT_STRUCTURAL_MATCH,
                                self.id,
                                self.span,
                                |lint| lint.build(&msg).emit(),
                            );
                        }
                        PatKind::Constant { value: cv }
                    } else {
                        if !self.saw_const_match_error.get() {
                            self.saw_const_match_error.set(true);
                            let msg = self.adt_derive_msg(adt_def);
                            if self.include_lint_checks {
                                tcx.sess.span_err(span, &msg);
                            } else {
                                tcx.sess.delay_span_bug(span, &msg)
                            }
                        }
                        PatKind::Wild
                    }
                }
                // All other references are converted into deref patterns and then recursively
                // convert the dereferenced constant to a pattern that is the sub-pattern of the
                // deref pattern.
                _ => {
                    let old = self.behind_reference.replace(true);
                    // In case there are structural-match violations somewhere in this subpattern,
                    // we fall back to a const pattern. If we do not do this, we may end up with
                    // a !structural-match constant that is not of reference type, which makes it
                    // very hard to invoke `PartialEq::eq` on it as a fallback.
                    let val = match self.recur(tcx.deref_const(self.param_env.and(cv)), false) {
                        Ok(subpattern) => PatKind::Deref { subpattern },
                        Err(_) => PatKind::Constant { value: cv },
                    };
                    self.behind_reference.set(old);
                    val
                }
            },
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::FnDef(..) => {
                PatKind::Constant { value: cv }
            }
            ty::RawPtr(pointee) if pointee.ty.is_sized(tcx.at(span), param_env) => {
                PatKind::Constant { value: cv }
            }
            // FIXME: these can have very suprising behaviour where optimization levels or other
            // compilation choices change the runtime behaviour of the match.
            // See https://github.com/rust-lang/rust/issues/70861 for examples.
            ty::FnPtr(..) | ty::RawPtr(..) => {
                if self.include_lint_checks
                    && !self.saw_const_match_error.get()
                    && !self.saw_const_match_lint.get()
                {
                    self.saw_const_match_lint.set(true);
                    let msg = "function pointers and unsized pointers in patterns behave \
                        unpredictably and should not be relied upon. \
                        See https://github.com/rust-lang/rust/issues/70861 for details.";
                    tcx.struct_span_lint_hir(
                        lint::builtin::POINTER_STRUCTURAL_MATCH,
                        id,
                        span,
                        |lint| lint.build(&msg).emit(),
                    );
                }
                PatKind::Constant { value: cv }
            }
            _ => {
                self.saw_const_match_error.set(true);
                let msg = format!("`{}` cannot be used in patterns", cv.ty);
                if self.include_lint_checks {
                    tcx.sess.span_err(span, &msg);
                } else {
                    tcx.sess.delay_span_bug(span, &msg)
                }
                PatKind::Wild
            }
        };

        if self.include_lint_checks
            && !self.saw_const_match_error.get()
            && !self.saw_const_match_lint.get()
            && mir_structural_match_violation
            // FIXME(#73448): Find a way to bring const qualification into parity with
            // `search_for_structural_match_violation` and then remove this condition.
            && self.search_for_structural_match_violation(cv.ty).is_some()
        {
            self.saw_const_match_lint.set(true);
            // Obtain the actual type that isn't annotated. If we just looked at `cv.ty` we
            // could get `Option<NonStructEq>`, even though `Option` is annotated with derive.
            let msg = self.search_for_structural_match_violation(cv.ty).unwrap().replace(
                "in a pattern,",
                "in a pattern, the constant's initializer must be trivial or",
            );
            tcx.struct_span_lint_hir(
                lint::builtin::NONTRIVIAL_STRUCTURAL_MATCH,
                id,
                span,
                |lint| lint.build(&msg).emit(),
            );
        }

        Ok(Pat { span, ty: cv.ty, kind: Box::new(kind) })
    }
}
