use core::ops::ControlFlow;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_apfloat::Float;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Diag;
use rustc_hir as hir;
use rustc_index::Idx;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::Obligation;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::thir::{FieldPat, Pat, PatKind};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitableExt, TypeVisitor, ValTree,
};
use rustc_middle::{mir, span_bug};
use rustc_span::def_id::DefId;
use rustc_span::{Span, sym};
use rustc_trait_selection::traits::ObligationCause;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use tracing::{debug, instrument, trace};

use super::PatCtxt;
use crate::errors::{
    ConstPatternDependsOnGenericParameter, CouldNotEvalConstPattern, InvalidPattern, NaNPattern,
    PointerPattern, TypeNotPartialEq, TypeNotStructural, UnionPattern, UnsizedPattern,
};

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
    /// Converts a constant to a pattern (if possible).
    /// This means aggregate values (like structs and enums) are converted
    /// to a pattern that matches the value (as if you'd compared via structural equality).
    ///
    /// Only type system constants are supported, as we are using valtrees
    /// as an intermediate step. Unfortunately those don't carry a type
    /// so we have to carry one ourselves.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn const_to_pat(
        &self,
        c: ty::Const<'tcx>,
        ty: Ty<'tcx>,
        id: hir::HirId,
        span: Span,
    ) -> Box<Pat<'tcx>> {
        let mut convert = ConstToPat::new(self, id, span, c);

        match c.kind() {
            ty::ConstKind::Unevaluated(uv) => convert.unevaluated_to_pat(uv, ty),
            ty::ConstKind::Value(cv) => convert.valtree_to_pat(cv.valtree, cv.ty),
            _ => span_bug!(span, "Invalid `ConstKind` for `const_to_pat`: {:?}", c),
        }
    }
}

struct ConstToPat<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    span: Span,
    id: hir::HirId,

    c: ty::Const<'tcx>,
}

impl<'tcx> ConstToPat<'tcx> {
    fn new(pat_ctxt: &PatCtxt<'_, 'tcx>, id: hir::HirId, span: Span, c: ty::Const<'tcx>) -> Self {
        trace!(?pat_ctxt.typeck_results.hir_owner);
        ConstToPat { tcx: pat_ctxt.tcx, typing_env: pat_ctxt.typing_env, span, id, c }
    }

    fn type_marked_structural(&self, ty: Ty<'tcx>) -> bool {
        ty.is_structural_eq_shallow(self.tcx)
    }

    /// We errored. Signal that in the pattern, so that follow up errors can be silenced.
    fn mk_err(&self, mut err: Diag<'_>, ty: Ty<'tcx>) -> Box<Pat<'tcx>> {
        if let ty::ConstKind::Unevaluated(uv) = self.c.kind() {
            let def_kind = self.tcx.def_kind(uv.def);
            if let hir::def::DefKind::AssocConst = def_kind
                && let Some(def_id) = uv.def.as_local()
            {
                // Include the container item in the output.
                err.span_label(self.tcx.def_span(self.tcx.local_parent(def_id)), "");
            }
            if let hir::def::DefKind::Const | hir::def::DefKind::AssocConst = def_kind {
                err.span_label(
                    self.tcx.def_span(uv.def),
                    crate::fluent_generated::mir_build_const_defined_here,
                );
            }
        }
        Box::new(Pat { span: self.span, ty, kind: PatKind::Error(err.emit()) })
    }

    fn unevaluated_to_pat(
        &mut self,
        uv: ty::UnevaluatedConst<'tcx>,
        ty: Ty<'tcx>,
    ) -> Box<Pat<'tcx>> {
        // It's not *technically* correct to be revealing opaque types here as borrowcheck has
        // not run yet. However, CTFE itself uses `TypingMode::PostAnalysis` unconditionally even
        // during typeck and not doing so has a lot of (undesirable) fallout (#101478, #119821).
        // As a result we always use a revealed env when resolving the instance to evaluate.
        //
        // FIXME: `const_eval_resolve_for_typeck` should probably just modify the env itself
        // instead of having this logic here
        let typing_env =
            self.tcx.erase_regions(self.typing_env).with_post_analysis_normalized(self.tcx);
        let uv = self.tcx.erase_regions(uv);

        // try to resolve e.g. associated constants to their definition on an impl, and then
        // evaluate the const.
        let valtree = match self.tcx.const_eval_resolve_for_typeck(typing_env, uv, self.span) {
            Ok(Ok(c)) => c,
            Err(ErrorHandled::Reported(_, _)) => {
                // Let's tell the use where this failing const occurs.
                let mut err =
                    self.tcx.dcx().create_err(CouldNotEvalConstPattern { span: self.span });
                // We've emitted an error on the original const, it would be redundant to complain
                // on its use as well.
                if let ty::ConstKind::Unevaluated(uv) = self.c.kind()
                    && let hir::def::DefKind::Const | hir::def::DefKind::AssocConst =
                        self.tcx.def_kind(uv.def)
                {
                    err.downgrade_to_delayed_bug();
                }
                return self.mk_err(err, ty);
            }
            Err(ErrorHandled::TooGeneric(_)) => {
                let mut e = self
                    .tcx
                    .dcx()
                    .create_err(ConstPatternDependsOnGenericParameter { span: self.span });
                for arg in uv.args {
                    if let ty::GenericArgKind::Type(ty) = arg.kind()
                        && let ty::Param(param_ty) = ty.kind()
                    {
                        let def_id = self.tcx.hir_enclosing_body_owner(self.id);
                        let generics = self.tcx.generics_of(def_id);
                        let param = generics.type_param(*param_ty, self.tcx);
                        let span = self.tcx.def_span(param.def_id);
                        e.span_label(span, "constant depends on this generic parameter");
                        if let Some(ident) = self.tcx.def_ident_span(def_id)
                            && self.tcx.sess.source_map().is_multiline(ident.between(span))
                        {
                            // Display the `fn` name as well in the diagnostic, as the generic isn't
                            // in the same line and it could be confusing otherwise.
                            e.span_label(ident, "");
                        }
                    }
                }
                return self.mk_err(e, ty);
            }
            Ok(Err(bad_ty)) => {
                // The pattern cannot be turned into a valtree.
                let e = match bad_ty.kind() {
                    ty::Adt(def, ..) => {
                        assert!(def.is_union());
                        self.tcx.dcx().create_err(UnionPattern { span: self.span })
                    }
                    ty::FnPtr(..) | ty::RawPtr(..) => {
                        self.tcx.dcx().create_err(PointerPattern { span: self.span })
                    }
                    _ => self.tcx.dcx().create_err(InvalidPattern {
                        span: self.span,
                        non_sm_ty: bad_ty,
                        prefix: bad_ty.prefix_string(self.tcx).to_string(),
                    }),
                };
                return self.mk_err(e, ty);
            }
        };

        // Convert the valtree to a const.
        let inlined_const_as_pat = self.valtree_to_pat(valtree, ty);

        if !inlined_const_as_pat.references_error() {
            // Always check for `PartialEq` if we had no other errors yet.
            if !type_has_partial_eq_impl(self.tcx, typing_env, ty).has_impl {
                let mut err = self.tcx.dcx().create_err(TypeNotPartialEq { span: self.span, ty });
                extend_type_not_partial_eq(self.tcx, typing_env, ty, &mut err);
                return self.mk_err(err, ty);
            }
        }

        // Wrap the pattern in a marker node to indicate that it is the result of lowering a
        // constant. This is used for diagnostics, and for unsafety checking of inline const blocks.
        let kind = PatKind::ExpandedConstant { subpattern: inlined_const_as_pat, def_id: uv.def };
        Box::new(Pat { kind, ty, span: self.span })
    }

    fn field_pats(
        &self,
        vals: impl Iterator<Item = (ValTree<'tcx>, Ty<'tcx>)>,
    ) -> Vec<FieldPat<'tcx>> {
        vals.enumerate()
            .map(|(idx, (val, ty))| {
                let field = FieldIdx::new(idx);
                // Patterns can only use monomorphic types.
                let ty = self.tcx.normalize_erasing_regions(self.typing_env, ty);
                FieldPat { field, pattern: *self.valtree_to_pat(val, ty) }
            })
            .collect()
    }

    // Recursive helper for `to_pat`; invoke that (instead of calling this directly).
    // FIXME(valtrees): Accept `ty::Value` instead of `Ty` and `ty::ValTree` separately.
    #[instrument(skip(self), level = "debug")]
    fn valtree_to_pat(&self, cv: ValTree<'tcx>, ty: Ty<'tcx>) -> Box<Pat<'tcx>> {
        let span = self.span;
        let tcx = self.tcx;
        let kind = match ty.kind() {
            ty::Adt(adt_def, _) if !self.type_marked_structural(ty) => {
                // Extremely important check for all ADTs! Make sure they opted-in to be used in
                // patterns.
                debug!("adt_def {:?} has !type_marked_structural for cv.ty: {:?}", adt_def, ty);
                let PartialEqImplStatus {
                    is_derived, structural_partial_eq, non_blanket_impl, ..
                } = type_has_partial_eq_impl(self.tcx, self.typing_env, ty);
                let (manual_partialeq_impl_span, manual_partialeq_impl_note) =
                    match (structural_partial_eq, non_blanket_impl) {
                        (true, _) => (None, false),
                        (_, Some(def_id)) if def_id.is_local() && !is_derived => {
                            (Some(tcx.def_span(def_id)), false)
                        }
                        _ => (None, true),
                    };
                let ty_def_span = tcx.def_span(adt_def.did());
                let err = TypeNotStructural {
                    span,
                    ty,
                    ty_def_span,
                    manual_partialeq_impl_span,
                    manual_partialeq_impl_note,
                };
                return self.mk_err(tcx.dcx().create_err(err), ty);
            }
            ty::Adt(adt_def, args) if adt_def.is_enum() => {
                let (&variant_index, fields) = cv.unwrap_branch().split_first().unwrap();
                let variant_index = VariantIdx::from_u32(variant_index.unwrap_leaf().to_u32());
                PatKind::Variant {
                    adt_def: *adt_def,
                    args,
                    variant_index,
                    subpatterns: self.field_pats(
                        fields.iter().copied().zip(
                            adt_def.variants()[variant_index]
                                .fields
                                .iter()
                                .map(|field| field.ty(tcx, args)),
                        ),
                    ),
                }
            }
            ty::Adt(def, args) => {
                assert!(!def.is_union()); // Valtree construction would never succeed for unions.
                PatKind::Leaf {
                    subpatterns: self.field_pats(cv.unwrap_branch().iter().copied().zip(
                        def.non_enum_variant().fields.iter().map(|field| field.ty(tcx, args)),
                    )),
                }
            }
            ty::Tuple(fields) => PatKind::Leaf {
                subpatterns: self.field_pats(cv.unwrap_branch().iter().copied().zip(fields.iter())),
            },
            ty::Slice(elem_ty) => PatKind::Slice {
                prefix: cv
                    .unwrap_branch()
                    .iter()
                    .map(|val| *self.valtree_to_pat(*val, *elem_ty))
                    .collect(),
                slice: None,
                suffix: Box::new([]),
            },
            ty::Array(elem_ty, _) => PatKind::Array {
                prefix: cv
                    .unwrap_branch()
                    .iter()
                    .map(|val| *self.valtree_to_pat(*val, *elem_ty))
                    .collect(),
                slice: None,
                suffix: Box::new([]),
            },
            ty::Str => {
                // String literal patterns may have type `str` if `deref_patterns` is enabled, in
                // order to allow `deref!("..."): String`. Since we need a `&str` for the comparison
                // when lowering to MIR in `Builder::perform_test`, treat the constant as a `&str`.
                // This works because `str` and `&str` have the same valtree representation.
                let ref_str_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, ty);
                PatKind::Constant {
                    value: mir::Const::Ty(ref_str_ty, ty::Const::new_value(tcx, cv, ref_str_ty)),
                }
            }
            ty::Ref(_, pointee_ty, ..) => match *pointee_ty.kind() {
                // `&str` is represented as a valtree, let's keep using this
                // optimization for now.
                ty::Str => PatKind::Constant {
                    value: mir::Const::Ty(ty, ty::Const::new_value(tcx, cv, ty)),
                },
                // All other references are converted into deref patterns and then recursively
                // convert the dereferenced constant to a pattern that is the sub-pattern of the
                // deref pattern.
                _ => {
                    if !pointee_ty.is_sized(tcx, self.typing_env) && !pointee_ty.is_slice() {
                        return self.mk_err(
                            tcx.dcx().create_err(UnsizedPattern { span, non_sm_ty: *pointee_ty }),
                            ty,
                        );
                    } else {
                        // References have the same valtree representation as their pointee.
                        PatKind::Deref { subpattern: self.valtree_to_pat(cv, *pointee_ty) }
                    }
                }
            },
            ty::Float(flt) => {
                let v = cv.unwrap_leaf();
                let is_nan = match flt {
                    ty::FloatTy::F16 => v.to_f16().is_nan(),
                    ty::FloatTy::F32 => v.to_f32().is_nan(),
                    ty::FloatTy::F64 => v.to_f64().is_nan(),
                    ty::FloatTy::F128 => v.to_f128().is_nan(),
                };
                if is_nan {
                    // NaNs are not ever equal to anything so they make no sense as patterns.
                    // Also see <https://github.com/rust-lang/rfcs/pull/3535>.
                    return self.mk_err(tcx.dcx().create_err(NaNPattern { span }), ty);
                } else {
                    PatKind::Constant {
                        value: mir::Const::Ty(ty, ty::Const::new_value(tcx, cv, ty)),
                    }
                }
            }
            ty::Pat(..) | ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::RawPtr(..) => {
                // The raw pointers we see here have been "vetted" by valtree construction to be
                // just integers, so we simply allow them.
                PatKind::Constant { value: mir::Const::Ty(ty, ty::Const::new_value(tcx, cv, ty)) }
            }
            ty::FnPtr(..) => {
                unreachable!(
                    "Valtree construction would never succeed for FnPtr, so this is unreachable."
                )
            }
            _ => {
                let err = InvalidPattern {
                    span,
                    non_sm_ty: ty,
                    prefix: ty.prefix_string(tcx).to_string(),
                };
                return self.mk_err(tcx.dcx().create_err(err), ty);
            }
        };

        Box::new(Pat { span, ty, kind })
    }
}

/// Given a type with type parameters, visit every ADT looking for types that need to
/// `#[derive(PartialEq)]` for it to be a structural type.
fn extend_type_not_partial_eq<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
    err: &mut Diag<'_>,
) {
    /// Collect all types that need to be `StructuralPartialEq`.
    struct UsedParamsNeedInstantiationVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        /// The user has written `impl PartialEq for Ty` which means it's non-structual.
        adts_with_manual_partialeq: FxHashSet<Span>,
        /// The type has no `PartialEq` implementation, neither manual or derived.
        adts_without_partialeq: FxHashSet<Span>,
        /// The user has written `impl PartialEq for Ty` which means it's non-structual,
        /// but we don't have a span to point at, so we'll just add them as a `note`.
        manual: FxHashSet<Ty<'tcx>>,
        /// The type has no `PartialEq` implementation, neither manual or derived, but
        /// we don't have a span to point at, so we'll just add them as a `note`.
        without: FxHashSet<Ty<'tcx>>,
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for UsedParamsNeedInstantiationVisitor<'tcx> {
        type Result = ControlFlow<()>;
        fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
            match ty.kind() {
                ty::Dynamic(..) => return ControlFlow::Break(()),
                // Unsafe binders never implement `PartialEq`, so avoid walking into them
                // which would require instantiating its binder with placeholders too.
                ty::UnsafeBinder(..) => return ControlFlow::Break(()),
                ty::FnPtr(..) => return ControlFlow::Continue(()),
                ty::Adt(def, _args) => {
                    let ty_def_id = def.did();
                    let ty_def_span = self.tcx.def_span(ty_def_id);
                    let PartialEqImplStatus {
                        has_impl,
                        is_derived,
                        structural_partial_eq,
                        non_blanket_impl,
                    } = type_has_partial_eq_impl(self.tcx, self.typing_env, ty);
                    match (has_impl, is_derived, structural_partial_eq, non_blanket_impl) {
                        (_, _, true, _) => {}
                        (true, false, _, Some(def_id)) if def_id.is_local() => {
                            self.adts_with_manual_partialeq.insert(self.tcx.def_span(def_id));
                        }
                        (true, false, _, _) if ty_def_id.is_local() => {
                            self.adts_with_manual_partialeq.insert(ty_def_span);
                        }
                        (false, _, _, _) if ty_def_id.is_local() => {
                            self.adts_without_partialeq.insert(ty_def_span);
                        }
                        (true, false, _, _) => {
                            self.manual.insert(ty);
                        }
                        (false, _, _, _) => {
                            self.without.insert(ty);
                        }
                        _ => {}
                    };
                    ty.super_visit_with(self)
                }
                _ => ty.super_visit_with(self),
            }
        }
    }
    let mut v = UsedParamsNeedInstantiationVisitor {
        tcx,
        typing_env,
        adts_with_manual_partialeq: FxHashSet::default(),
        adts_without_partialeq: FxHashSet::default(),
        manual: FxHashSet::default(),
        without: FxHashSet::default(),
    };
    if v.visit_ty(ty).is_break() {
        return;
    }
    #[allow(rustc::potential_query_instability)] // Span labels will be sorted by the rendering
    for span in v.adts_with_manual_partialeq {
        err.span_note(span, "the `PartialEq` trait must be derived, manual `impl`s are not sufficient; see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details");
    }
    #[allow(rustc::potential_query_instability)] // Span labels will be sorted by the rendering
    for span in v.adts_without_partialeq {
        err.span_label(
            span,
            "must be annotated with `#[derive(PartialEq)]` to be usable in patterns",
        );
    }
    #[allow(rustc::potential_query_instability)]
    let mut manual: Vec<_> = v.manual.into_iter().map(|t| t.to_string()).collect();
    manual.sort();
    for ty in manual {
        err.note(format!(
            "`{ty}` must be annotated with `#[derive(PartialEq)]` to be usable in patterns, manual `impl`s are not sufficient; see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details"
        ));
    }
    #[allow(rustc::potential_query_instability)]
    let mut without: Vec<_> = v.without.into_iter().map(|t| t.to_string()).collect();
    without.sort();
    for ty in without {
        err.note(format!(
            "`{ty}` must be annotated with `#[derive(PartialEq)]` to be usable in patterns"
        ));
    }
}

#[derive(Debug)]
struct PartialEqImplStatus {
    has_impl: bool,
    is_derived: bool,
    structural_partial_eq: bool,
    non_blanket_impl: Option<DefId>,
}

#[instrument(level = "trace", skip(tcx), ret)]
fn type_has_partial_eq_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> PartialEqImplStatus {
    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    // double-check there even *is* a semantic `PartialEq` to dispatch to.
    //
    // (If there isn't, then we can safely issue a hard
    // error, because that's never worked, due to compiler
    // using `PartialEq::eq` in this scenario in the past.)
    let partial_eq_trait_id = tcx.require_lang_item(hir::LangItem::PartialEq, None);
    let structural_partial_eq_trait_id = tcx.require_lang_item(hir::LangItem::StructuralPeq, None);

    let partial_eq_obligation = Obligation::new(
        tcx,
        ObligationCause::dummy(),
        param_env,
        ty::TraitRef::new(tcx, partial_eq_trait_id, [ty, ty]),
    );

    let mut automatically_derived = false;
    let mut structural_peq = false;
    let mut impl_def_id = None;
    for def_id in tcx.non_blanket_impls_for_ty(partial_eq_trait_id, ty) {
        automatically_derived = tcx.has_attr(def_id, sym::automatically_derived);
        impl_def_id = Some(def_id);
    }
    for _ in tcx.non_blanket_impls_for_ty(structural_partial_eq_trait_id, ty) {
        structural_peq = true;
    }
    // This *could* accept a type that isn't actually `PartialEq`, because region bounds get
    // ignored. However that should be pretty much impossible since consts that do not depend on
    // generics can only mention the `'static` lifetime, and how would one have a type that's
    // `PartialEq` for some lifetime but *not* for `'static`? If this ever becomes a problem
    // we'll need to leave some sort of trace of this requirement in the MIR so that borrowck
    // can ensure that the type really implements `PartialEq`.
    // We also do *not* require `const PartialEq`, not even in `const fn`. This violates the model
    // that patterns can only do things that the code could also do without patterns, but it is
    // needed for backwards compatibility. The actual pattern matching compares primitive values,
    // `PartialEq::eq` never gets invoked, so there's no risk of us running non-const code.
    PartialEqImplStatus {
        has_impl: infcx.predicate_must_hold_modulo_regions(&partial_eq_obligation),
        is_derived: automatically_derived,
        structural_partial_eq: structural_peq,
        non_blanket_impl: impl_def_id,
    }
}
