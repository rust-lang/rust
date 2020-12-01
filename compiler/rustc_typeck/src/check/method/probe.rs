use super::suggest;
use super::MethodError;
use super::NoMatchData;
use super::{CandidateSource, ImplSource, TraitSource};

use crate::check::FnCtxt;
use crate::errors::MethodCallOnUnknownType;
use crate::hir::def::DefKind;
use crate::hir::def_id::DefId;

use rustc_ast as ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def::Namespace;
use rustc_infer::infer::canonical::OriginalQueryValues;
use rustc_infer::infer::canonical::{Canonical, QueryResponse};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_infer::infer::{self, InferOk, TyCtxtInferExt};
use rustc_middle::middle::stability;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::GenericParamDefKind;
use rustc_middle::ty::{
    self, ParamEnvAnd, ToPolyTraitRef, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness,
};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::lev_distance::{find_best_match_for_name, lev_distance};
use rustc_span::{symbol::Ident, Span, Symbol, DUMMY_SP};
use rustc_trait_selection::autoderef::{self, Autoderef};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::query::method_autoderef::MethodAutoderefBadTy;
use rustc_trait_selection::traits::query::method_autoderef::{
    CandidateStep, MethodAutoderefStepsResult,
};
use rustc_trait_selection::traits::query::CanonicalTyGoal;
use rustc_trait_selection::traits::{self, ObligationCause};
use std::cmp::max;
use std::iter;
use std::mem;
use std::ops::Deref;

use smallvec::{smallvec, SmallVec};

use self::CandidateKind::*;
pub use self::PickKind::*;

/// Boolean flag used to indicate if this search is for a suggestion
/// or not. If true, we can allow ambiguity and so forth.
#[derive(Clone, Copy)]
pub struct IsSuggestion(pub bool);

struct ProbeContext<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    mode: Mode,
    method_name: Option<Ident>,
    return_type: Option<Ty<'tcx>>,

    /// This is the OriginalQueryValues for the steps queries
    /// that are answered in steps.
    orig_steps_var_values: OriginalQueryValues<'tcx>,
    steps: Lrc<Vec<CandidateStep<'tcx>>>,

    inherent_candidates: Vec<Candidate<'tcx>>,
    extension_candidates: Vec<Candidate<'tcx>>,
    impl_dups: FxHashSet<DefId>,

    /// Collects near misses when the candidate functions are missing a `self` keyword and is only
    /// used for error reporting
    static_candidates: Vec<CandidateSource>,

    /// When probing for names, include names that are close to the
    /// requested name (by Levensthein distance)
    allow_similar_names: bool,

    /// Some(candidate) if there is a private candidate
    private_candidate: Option<(DefKind, DefId)>,

    /// Collects near misses when trait bounds for type parameters are unsatisfied and is only used
    /// for error reporting
    unsatisfied_predicates: Vec<(ty::Predicate<'tcx>, Option<ty::Predicate<'tcx>>)>,

    is_suggestion: IsSuggestion,
}

impl<'a, 'tcx> Deref for ProbeContext<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

#[derive(Debug)]
struct Candidate<'tcx> {
    // Candidates are (I'm not quite sure, but they are mostly) basically
    // some metadata on top of a `ty::AssocItem` (without substs).
    //
    // However, method probing wants to be able to evaluate the predicates
    // for a function with the substs applied - for example, if a function
    // has `where Self: Sized`, we don't want to consider it unless `Self`
    // is actually `Sized`, and similarly, return-type suggestions want
    // to consider the "actual" return type.
    //
    // The way this is handled is through `xform_self_ty`. It contains
    // the receiver type of this candidate, but `xform_self_ty`,
    // `xform_ret_ty` and `kind` (which contains the predicates) have the
    // generic parameters of this candidate substituted with the *same set*
    // of inference variables, which acts as some weird sort of "query".
    //
    // When we check out a candidate, we require `xform_self_ty` to be
    // a subtype of the passed-in self-type, and this equates the type
    // variables in the rest of the fields.
    //
    // For example, if we have this candidate:
    // ```
    //    trait Foo {
    //        fn foo(&self) where Self: Sized;
    //    }
    // ```
    //
    // Then `xform_self_ty` will be `&'erased ?X` and `kind` will contain
    // the predicate `?X: Sized`, so if we are evaluating `Foo` for a
    // the receiver `&T`, we'll do the subtyping which will make `?X`
    // get the right value, then when we evaluate the predicate we'll check
    // if `T: Sized`.
    xform_self_ty: Ty<'tcx>,
    xform_ret_ty: Option<Ty<'tcx>>,
    item: ty::AssocItem,
    kind: CandidateKind<'tcx>,
    import_ids: SmallVec<[LocalDefId; 1]>,
}

#[derive(Debug)]
enum CandidateKind<'tcx> {
    InherentImplCandidate(
        SubstsRef<'tcx>,
        // Normalize obligations
        Vec<traits::PredicateObligation<'tcx>>,
    ),
    ObjectCandidate,
    TraitCandidate(ty::TraitRef<'tcx>),
    WhereClauseCandidate(
        // Trait
        ty::PolyTraitRef<'tcx>,
    ),
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum ProbeResult {
    NoMatch,
    BadReturnType,
    Match,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Pick<'tcx> {
    pub item: ty::AssocItem,
    pub kind: PickKind<'tcx>,
    pub import_ids: SmallVec<[LocalDefId; 1]>,

    // Indicates that the source expression should be autoderef'd N times
    //
    // A = expr | *expr | **expr | ...
    pub autoderefs: usize,

    // Indicates that an autoref is applied after the optional autoderefs
    //
    // B = A | &A | &mut A
    pub autoref: Option<hir::Mutability>,

    // Indicates that the source expression should be "unsized" to a
    // target type. This should probably eventually go away in favor
    // of just coercing method receivers.
    //
    // C = B | unsize(B)
    pub unsize: Option<Ty<'tcx>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PickKind<'tcx> {
    InherentImplPick,
    ObjectPick,
    TraitPick,
    WhereClausePick(
        // Trait
        ty::PolyTraitRef<'tcx>,
    ),
}

pub type PickResult<'tcx> = Result<Pick<'tcx>, MethodError<'tcx>>;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Mode {
    // An expression of the form `receiver.method_name(...)`.
    // Autoderefs are performed on `receiver`, lookup is done based on the
    // `self` argument  of the method, and static methods aren't considered.
    MethodCall,
    // An expression of the form `Type::item` or `<T>::item`.
    // No autoderefs are performed, lookup is done based on the type each
    // implementation is for, and static methods are included.
    Path,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum ProbeScope {
    // Assemble candidates coming only from traits in scope.
    TraitsInScope,

    // Assemble candidates coming from all traits.
    AllTraits,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// This is used to offer suggestions to users. It returns methods
    /// that could have been called which have the desired return
    /// type. Some effort is made to rule out methods that, if called,
    /// would result in an error (basically, the same criteria we
    /// would use to decide if a method is a plausible fit for
    /// ambiguity purposes).
    pub fn probe_for_return_type(
        &self,
        span: Span,
        mode: Mode,
        return_type: Ty<'tcx>,
        self_ty: Ty<'tcx>,
        scope_expr_id: hir::HirId,
    ) -> Vec<ty::AssocItem> {
        debug!(
            "probe(self_ty={:?}, return_type={}, scope_expr_id={})",
            self_ty, return_type, scope_expr_id
        );
        let method_names = self
            .probe_op(
                span,
                mode,
                None,
                Some(return_type),
                IsSuggestion(true),
                self_ty,
                scope_expr_id,
                ProbeScope::AllTraits,
                |probe_cx| Ok(probe_cx.candidate_method_names()),
            )
            .unwrap_or_default();
        method_names
            .iter()
            .flat_map(|&method_name| {
                self.probe_op(
                    span,
                    mode,
                    Some(method_name),
                    Some(return_type),
                    IsSuggestion(true),
                    self_ty,
                    scope_expr_id,
                    ProbeScope::AllTraits,
                    |probe_cx| probe_cx.pick(),
                )
                .ok()
                .map(|pick| pick.item)
            })
            .collect()
    }

    pub fn probe_for_name(
        &self,
        span: Span,
        mode: Mode,
        item_name: Ident,
        is_suggestion: IsSuggestion,
        self_ty: Ty<'tcx>,
        scope_expr_id: hir::HirId,
        scope: ProbeScope,
    ) -> PickResult<'tcx> {
        debug!(
            "probe(self_ty={:?}, item_name={}, scope_expr_id={})",
            self_ty, item_name, scope_expr_id
        );
        self.probe_op(
            span,
            mode,
            Some(item_name),
            None,
            is_suggestion,
            self_ty,
            scope_expr_id,
            scope,
            |probe_cx| probe_cx.pick(),
        )
    }

    fn probe_op<OP, R>(
        &'a self,
        span: Span,
        mode: Mode,
        method_name: Option<Ident>,
        return_type: Option<Ty<'tcx>>,
        is_suggestion: IsSuggestion,
        self_ty: Ty<'tcx>,
        scope_expr_id: hir::HirId,
        scope: ProbeScope,
        op: OP,
    ) -> Result<R, MethodError<'tcx>>
    where
        OP: FnOnce(ProbeContext<'a, 'tcx>) -> Result<R, MethodError<'tcx>>,
    {
        let mut orig_values = OriginalQueryValues::default();
        let param_env_and_self_ty = self.infcx.canonicalize_query(
            ParamEnvAnd { param_env: self.param_env, value: self_ty },
            &mut orig_values,
        );

        let steps = if mode == Mode::MethodCall {
            self.tcx.method_autoderef_steps(param_env_and_self_ty)
        } else {
            self.infcx.probe(|_| {
                // Mode::Path - the deref steps is "trivial". This turns
                // our CanonicalQuery into a "trivial" QueryResponse. This
                // is a bit inefficient, but I don't think that writing
                // special handling for this "trivial case" is a good idea.

                let infcx = &self.infcx;
                let (ParamEnvAnd { param_env: _, value: self_ty }, canonical_inference_vars) =
                    infcx.instantiate_canonical_with_fresh_inference_vars(
                        span,
                        &param_env_and_self_ty,
                    );
                debug!(
                    "probe_op: Mode::Path, param_env_and_self_ty={:?} self_ty={:?}",
                    param_env_and_self_ty, self_ty
                );
                MethodAutoderefStepsResult {
                    steps: Lrc::new(vec![CandidateStep {
                        self_ty: self.make_query_response_ignoring_pending_obligations(
                            canonical_inference_vars,
                            self_ty,
                        ),
                        autoderefs: 0,
                        from_unsafe_deref: false,
                        unsize: false,
                    }]),
                    opt_bad_ty: None,
                    reached_recursion_limit: false,
                }
            })
        };

        // If our autoderef loop had reached the recursion limit,
        // report an overflow error, but continue going on with
        // the truncated autoderef list.
        if steps.reached_recursion_limit {
            self.probe(|_| {
                let ty = &steps
                    .steps
                    .last()
                    .unwrap_or_else(|| span_bug!(span, "reached the recursion limit in 0 steps?"))
                    .self_ty;
                let ty = self
                    .probe_instantiate_query_response(span, &orig_values, ty)
                    .unwrap_or_else(|_| span_bug!(span, "instantiating {:?} failed?", ty));
                autoderef::report_autoderef_recursion_limit_error(self.tcx, span, ty.value);
            });
        }

        // If we encountered an `_` type or an error type during autoderef, this is
        // ambiguous.
        if let Some(bad_ty) = &steps.opt_bad_ty {
            if is_suggestion.0 {
                // Ambiguity was encountered during a suggestion. Just keep going.
                debug!("ProbeContext: encountered ambiguity in suggestion");
            } else if bad_ty.reached_raw_pointer && !self.tcx.features().arbitrary_self_types {
                // this case used to be allowed by the compiler,
                // so we do a future-compat lint here for the 2015 edition
                // (see https://github.com/rust-lang/rust/issues/46906)
                if self.tcx.sess.rust_2018() {
                    self.tcx.sess.emit_err(MethodCallOnUnknownType { span });
                } else {
                    self.tcx.struct_span_lint_hir(
                        lint::builtin::TYVAR_BEHIND_RAW_POINTER,
                        scope_expr_id,
                        span,
                        |lint| lint.build("type annotations needed").emit(),
                    );
                }
            } else {
                // Encountered a real ambiguity, so abort the lookup. If `ty` is not
                // an `Err`, report the right "type annotations needed" error pointing
                // to it.
                let ty = &bad_ty.ty;
                let ty = self
                    .probe_instantiate_query_response(span, &orig_values, ty)
                    .unwrap_or_else(|_| span_bug!(span, "instantiating {:?} failed?", ty));
                let ty = self.structurally_resolved_type(span, ty.value);
                assert!(matches!(ty.kind(), ty::Error(_)));
                return Err(MethodError::NoMatch(NoMatchData::new(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    None,
                    mode,
                )));
            }
        }

        debug!("ProbeContext: steps for self_ty={:?} are {:?}", self_ty, steps);

        // this creates one big transaction so that all type variables etc
        // that we create during the probe process are removed later
        self.probe(|_| {
            let mut probe_cx = ProbeContext::new(
                self,
                span,
                mode,
                method_name,
                return_type,
                orig_values,
                steps.steps,
                is_suggestion,
            );

            probe_cx.assemble_inherent_candidates();
            match scope {
                ProbeScope::TraitsInScope => {
                    probe_cx.assemble_extension_candidates_for_traits_in_scope(scope_expr_id)?
                }
                ProbeScope::AllTraits => probe_cx.assemble_extension_candidates_for_all_traits()?,
            };
            op(probe_cx)
        })
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    providers.method_autoderef_steps = method_autoderef_steps;
}

fn method_autoderef_steps<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: CanonicalTyGoal<'tcx>,
) -> MethodAutoderefStepsResult<'tcx> {
    debug!("method_autoderef_steps({:?})", goal);

    tcx.infer_ctxt().enter_with_canonical(DUMMY_SP, &goal, |ref infcx, goal, inference_vars| {
        let ParamEnvAnd { param_env, value: self_ty } = goal;

        let mut autoderef =
            Autoderef::new(infcx, param_env, hir::CRATE_HIR_ID, DUMMY_SP, self_ty, DUMMY_SP)
                .include_raw_pointers()
                .silence_errors();
        let mut reached_raw_pointer = false;
        let mut steps: Vec<_> = autoderef
            .by_ref()
            .map(|(ty, d)| {
                let step = CandidateStep {
                    self_ty: infcx.make_query_response_ignoring_pending_obligations(
                        inference_vars.clone(),
                        ty,
                    ),
                    autoderefs: d,
                    from_unsafe_deref: reached_raw_pointer,
                    unsize: false,
                };
                if let ty::RawPtr(_) = ty.kind() {
                    // all the subsequent steps will be from_unsafe_deref
                    reached_raw_pointer = true;
                }
                step
            })
            .collect();

        let final_ty = autoderef.final_ty(true);
        let opt_bad_ty = match final_ty.kind() {
            ty::Infer(ty::TyVar(_)) | ty::Error(_) => Some(MethodAutoderefBadTy {
                reached_raw_pointer,
                ty: infcx
                    .make_query_response_ignoring_pending_obligations(inference_vars, final_ty),
            }),
            ty::Array(elem_ty, _) => {
                let dereferences = steps.len() - 1;

                steps.push(CandidateStep {
                    self_ty: infcx.make_query_response_ignoring_pending_obligations(
                        inference_vars,
                        infcx.tcx.mk_slice(elem_ty),
                    ),
                    autoderefs: dereferences,
                    // this could be from an unsafe deref if we had
                    // a *mut/const [T; N]
                    from_unsafe_deref: reached_raw_pointer,
                    unsize: true,
                });

                None
            }
            _ => None,
        };

        debug!("method_autoderef_steps: steps={:?} opt_bad_ty={:?}", steps, opt_bad_ty);

        MethodAutoderefStepsResult {
            steps: Lrc::new(steps),
            opt_bad_ty: opt_bad_ty.map(Lrc::new),
            reached_recursion_limit: autoderef.reached_recursion_limit(),
        }
    })
}

impl<'a, 'tcx> ProbeContext<'a, 'tcx> {
    fn new(
        fcx: &'a FnCtxt<'a, 'tcx>,
        span: Span,
        mode: Mode,
        method_name: Option<Ident>,
        return_type: Option<Ty<'tcx>>,
        orig_steps_var_values: OriginalQueryValues<'tcx>,
        steps: Lrc<Vec<CandidateStep<'tcx>>>,
        is_suggestion: IsSuggestion,
    ) -> ProbeContext<'a, 'tcx> {
        ProbeContext {
            fcx,
            span,
            mode,
            method_name,
            return_type,
            inherent_candidates: Vec::new(),
            extension_candidates: Vec::new(),
            impl_dups: FxHashSet::default(),
            orig_steps_var_values,
            steps,
            static_candidates: Vec::new(),
            allow_similar_names: false,
            private_candidate: None,
            unsatisfied_predicates: Vec::new(),
            is_suggestion,
        }
    }

    fn reset(&mut self) {
        self.inherent_candidates.clear();
        self.extension_candidates.clear();
        self.impl_dups.clear();
        self.static_candidates.clear();
        self.private_candidate = None;
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY

    fn push_candidate(&mut self, candidate: Candidate<'tcx>, is_inherent: bool) {
        let is_accessible = if let Some(name) = self.method_name {
            let item = candidate.item;
            let def_scope =
                self.tcx.adjust_ident_and_get_scope(name, item.container.id(), self.body_id).1;
            item.vis.is_accessible_from(def_scope, self.tcx)
        } else {
            true
        };
        if is_accessible {
            if is_inherent {
                self.inherent_candidates.push(candidate);
            } else {
                self.extension_candidates.push(candidate);
            }
        } else if self.private_candidate.is_none() {
            self.private_candidate =
                Some((candidate.item.kind.as_def_kind(), candidate.item.def_id));
        }
    }

    fn assemble_inherent_candidates(&mut self) {
        let steps = Lrc::clone(&self.steps);
        for step in steps.iter() {
            self.assemble_probe(&step.self_ty);
        }
    }

    fn assemble_probe(&mut self, self_ty: &Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>) {
        debug!("assemble_probe: self_ty={:?}", self_ty);
        let lang_items = self.tcx.lang_items();

        match *self_ty.value.value.kind() {
            ty::Dynamic(ref data, ..) => {
                if let Some(p) = data.principal() {
                    // Subtle: we can't use `instantiate_query_response` here: using it will
                    // commit to all of the type equalities assumed by inference going through
                    // autoderef (see the `method-probe-no-guessing` test).
                    //
                    // However, in this code, it is OK if we end up with an object type that is
                    // "more general" than the object type that we are evaluating. For *every*
                    // object type `MY_OBJECT`, a function call that goes through a trait-ref
                    // of the form `<MY_OBJECT as SuperTraitOf(MY_OBJECT)>::func` is a valid
                    // `ObjectCandidate`, and it should be discoverable "exactly" through one
                    // of the iterations in the autoderef loop, so there is no problem with it
                    // being discoverable in another one of these iterations.
                    //
                    // Using `instantiate_canonical_with_fresh_inference_vars` on our
                    // `Canonical<QueryResponse<Ty<'tcx>>>` and then *throwing away* the
                    // `CanonicalVarValues` will exactly give us such a generalization - it
                    // will still match the original object type, but it won't pollute our
                    // type variables in any form, so just do that!
                    let (QueryResponse { value: generalized_self_ty, .. }, _ignored_var_values) =
                        self.fcx
                            .instantiate_canonical_with_fresh_inference_vars(self.span, &self_ty);

                    self.assemble_inherent_candidates_from_object(generalized_self_ty);
                    self.assemble_inherent_impl_candidates_for_type(p.def_id());
                }
            }
            ty::Adt(def, _) => {
                self.assemble_inherent_impl_candidates_for_type(def.did);
            }
            ty::Foreign(did) => {
                self.assemble_inherent_impl_candidates_for_type(did);
            }
            ty::Param(p) => {
                self.assemble_inherent_candidates_from_param(p);
            }
            ty::Bool => {
                let lang_def_id = lang_items.bool_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::Char => {
                let lang_def_id = lang_items.char_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::Str => {
                let lang_def_id = lang_items.str_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);

                let lang_def_id = lang_items.str_alloc_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::Slice(_) => {
                for &lang_def_id in &[
                    lang_items.slice_impl(),
                    lang_items.slice_u8_impl(),
                    lang_items.slice_alloc_impl(),
                    lang_items.slice_u8_alloc_impl(),
                ] {
                    self.assemble_inherent_impl_for_primitive(lang_def_id);
                }
            }
            ty::Array(_, _) => {
                let lang_def_id = lang_items.array_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::RawPtr(ty::TypeAndMut { ty: _, mutbl }) => {
                let (lang_def_id1, lang_def_id2) = match mutbl {
                    hir::Mutability::Not => {
                        (lang_items.const_ptr_impl(), lang_items.const_slice_ptr_impl())
                    }
                    hir::Mutability::Mut => {
                        (lang_items.mut_ptr_impl(), lang_items.mut_slice_ptr_impl())
                    }
                };
                self.assemble_inherent_impl_for_primitive(lang_def_id1);
                self.assemble_inherent_impl_for_primitive(lang_def_id2);
            }
            ty::Int(i) => {
                let lang_def_id = match i {
                    ast::IntTy::I8 => lang_items.i8_impl(),
                    ast::IntTy::I16 => lang_items.i16_impl(),
                    ast::IntTy::I32 => lang_items.i32_impl(),
                    ast::IntTy::I64 => lang_items.i64_impl(),
                    ast::IntTy::I128 => lang_items.i128_impl(),
                    ast::IntTy::Isize => lang_items.isize_impl(),
                };
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::Uint(i) => {
                let lang_def_id = match i {
                    ast::UintTy::U8 => lang_items.u8_impl(),
                    ast::UintTy::U16 => lang_items.u16_impl(),
                    ast::UintTy::U32 => lang_items.u32_impl(),
                    ast::UintTy::U64 => lang_items.u64_impl(),
                    ast::UintTy::U128 => lang_items.u128_impl(),
                    ast::UintTy::Usize => lang_items.usize_impl(),
                };
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::Float(f) => {
                let (lang_def_id1, lang_def_id2) = match f {
                    ast::FloatTy::F32 => (lang_items.f32_impl(), lang_items.f32_runtime_impl()),
                    ast::FloatTy::F64 => (lang_items.f64_impl(), lang_items.f64_runtime_impl()),
                };
                self.assemble_inherent_impl_for_primitive(lang_def_id1);
                self.assemble_inherent_impl_for_primitive(lang_def_id2);
            }
            _ => {}
        }
    }

    fn assemble_inherent_impl_for_primitive(&mut self, lang_def_id: Option<DefId>) {
        if let Some(impl_def_id) = lang_def_id {
            self.assemble_inherent_impl_probe(impl_def_id);
        }
    }

    fn assemble_inherent_impl_candidates_for_type(&mut self, def_id: DefId) {
        let impl_def_ids = self.tcx.at(self.span).inherent_impls(def_id);
        for &impl_def_id in impl_def_ids.iter() {
            self.assemble_inherent_impl_probe(impl_def_id);
        }
    }

    fn assemble_inherent_impl_probe(&mut self, impl_def_id: DefId) {
        if !self.impl_dups.insert(impl_def_id) {
            return; // already visited
        }

        debug!("assemble_inherent_impl_probe {:?}", impl_def_id);

        for item in self.impl_or_trait_item(impl_def_id) {
            if !self.has_applicable_self(&item) {
                // No receiver declared. Not a candidate.
                self.record_static_candidate(ImplSource(impl_def_id));
                continue;
            }

            let (impl_ty, impl_substs) = self.impl_ty_and_substs(impl_def_id);
            let impl_ty = impl_ty.subst(self.tcx, impl_substs);

            // Determine the receiver type that the method itself expects.
            let xform_tys = self.xform_self_ty(&item, impl_ty, impl_substs);

            // We can't use normalize_associated_types_in as it will pollute the
            // fcx's fulfillment context after this probe is over.
            let cause = traits::ObligationCause::misc(self.span, self.body_id);
            let selcx = &mut traits::SelectionContext::new(self.fcx);
            let traits::Normalized { value: (xform_self_ty, xform_ret_ty), obligations } =
                traits::normalize(selcx, self.param_env, cause, xform_tys);
            debug!(
                "assemble_inherent_impl_probe: xform_self_ty = {:?}/{:?}",
                xform_self_ty, xform_ret_ty
            );

            self.push_candidate(
                Candidate {
                    xform_self_ty,
                    xform_ret_ty,
                    item,
                    kind: InherentImplCandidate(impl_substs, obligations),
                    import_ids: smallvec![],
                },
                true,
            );
        }
    }

    fn assemble_inherent_candidates_from_object(&mut self, self_ty: Ty<'tcx>) {
        debug!("assemble_inherent_candidates_from_object(self_ty={:?})", self_ty);

        let principal = match self_ty.kind() {
            ty::Dynamic(ref data, ..) => Some(data),
            _ => None,
        }
        .and_then(|data| data.principal())
        .unwrap_or_else(|| {
            span_bug!(
                self.span,
                "non-object {:?} in assemble_inherent_candidates_from_object",
                self_ty
            )
        });

        // It is illegal to invoke a method on a trait instance that
        // refers to the `Self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use
        // a substitution that replaces `Self` with the object type
        // itself. Hence, a `&self` method will wind up with an
        // argument type like `&Trait`.
        let trait_ref = principal.with_self_ty(self.tcx, self_ty);
        self.elaborate_bounds(iter::once(trait_ref), |this, new_trait_ref, item| {
            let new_trait_ref = this.erase_late_bound_regions(new_trait_ref);

            let (xform_self_ty, xform_ret_ty) =
                this.xform_self_ty(&item, new_trait_ref.self_ty(), new_trait_ref.substs);
            this.push_candidate(
                Candidate {
                    xform_self_ty,
                    xform_ret_ty,
                    item,
                    kind: ObjectCandidate,
                    import_ids: smallvec![],
                },
                true,
            );
        });
    }

    fn assemble_inherent_candidates_from_param(&mut self, param_ty: ty::ParamTy) {
        // FIXME: do we want to commit to this behavior for param bounds?
        debug!("assemble_inherent_candidates_from_param(param_ty={:?})", param_ty);

        let bounds = self.param_env.caller_bounds().iter().filter_map(|predicate| {
            let bound_predicate = predicate.bound_atom();
            match bound_predicate.skip_binder() {
                ty::PredicateAtom::Trait(trait_predicate, _) => {
                    match *trait_predicate.trait_ref.self_ty().kind() {
                        ty::Param(p) if p == param_ty => {
                            Some(bound_predicate.rebind(trait_predicate.trait_ref))
                        }
                        _ => None,
                    }
                }
                ty::PredicateAtom::Subtype(..)
                | ty::PredicateAtom::Projection(..)
                | ty::PredicateAtom::RegionOutlives(..)
                | ty::PredicateAtom::WellFormed(..)
                | ty::PredicateAtom::ObjectSafe(..)
                | ty::PredicateAtom::ClosureKind(..)
                | ty::PredicateAtom::TypeOutlives(..)
                | ty::PredicateAtom::ConstEvaluatable(..)
                | ty::PredicateAtom::ConstEquate(..)
                | ty::PredicateAtom::TypeWellFormedFromEnv(..) => None,
            }
        });

        self.elaborate_bounds(bounds, |this, poly_trait_ref, item| {
            let trait_ref = this.erase_late_bound_regions(poly_trait_ref);

            let (xform_self_ty, xform_ret_ty) =
                this.xform_self_ty(&item, trait_ref.self_ty(), trait_ref.substs);

            // Because this trait derives from a where-clause, it
            // should not contain any inference variables or other
            // artifacts. This means it is safe to put into the
            // `WhereClauseCandidate` and (eventually) into the
            // `WhereClausePick`.
            assert!(!trait_ref.substs.needs_infer());

            this.push_candidate(
                Candidate {
                    xform_self_ty,
                    xform_ret_ty,
                    item,
                    kind: WhereClauseCandidate(poly_trait_ref),
                    import_ids: smallvec![],
                },
                true,
            );
        });
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn elaborate_bounds<F>(
        &mut self,
        bounds: impl Iterator<Item = ty::PolyTraitRef<'tcx>>,
        mut mk_cand: F,
    ) where
        F: for<'b> FnMut(&mut ProbeContext<'b, 'tcx>, ty::PolyTraitRef<'tcx>, ty::AssocItem),
    {
        let tcx = self.tcx;
        for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
            debug!("elaborate_bounds(bound_trait_ref={:?})", bound_trait_ref);
            for item in self.impl_or_trait_item(bound_trait_ref.def_id()) {
                if !self.has_applicable_self(&item) {
                    self.record_static_candidate(TraitSource(bound_trait_ref.def_id()));
                } else {
                    mk_cand(self, bound_trait_ref, item);
                }
            }
        }
    }

    fn assemble_extension_candidates_for_traits_in_scope(
        &mut self,
        expr_hir_id: hir::HirId,
    ) -> Result<(), MethodError<'tcx>> {
        let mut duplicates = FxHashSet::default();
        let opt_applicable_traits = self.tcx.in_scope_traits(expr_hir_id);
        if let Some(applicable_traits) = opt_applicable_traits {
            for trait_candidate in applicable_traits.iter() {
                let trait_did = trait_candidate.def_id;
                if duplicates.insert(trait_did) {
                    let result = self.assemble_extension_candidates_for_trait(
                        &trait_candidate.import_ids,
                        trait_did,
                    );
                    result?;
                }
            }
        }
        Ok(())
    }

    fn assemble_extension_candidates_for_all_traits(&mut self) -> Result<(), MethodError<'tcx>> {
        let mut duplicates = FxHashSet::default();
        for trait_info in suggest::all_traits(self.tcx) {
            if duplicates.insert(trait_info.def_id) {
                self.assemble_extension_candidates_for_trait(&smallvec![], trait_info.def_id)?;
            }
        }
        Ok(())
    }

    pub fn matches_return_type(
        &self,
        method: &ty::AssocItem,
        self_ty: Option<Ty<'tcx>>,
        expected: Ty<'tcx>,
    ) -> bool {
        match method.kind {
            ty::AssocKind::Fn => {
                let fty = self.tcx.fn_sig(method.def_id);
                self.probe(|_| {
                    let substs = self.fresh_substs_for_item(self.span, method.def_id);
                    let fty = fty.subst(self.tcx, substs);
                    let (fty, _) =
                        self.replace_bound_vars_with_fresh_vars(self.span, infer::FnCall, fty);

                    if let Some(self_ty) = self_ty {
                        if self
                            .at(&ObligationCause::dummy(), self.param_env)
                            .sup(fty.inputs()[0], self_ty)
                            .is_err()
                        {
                            return false;
                        }
                    }
                    self.can_sub(self.param_env, fty.output(), expected).is_ok()
                })
            }
            _ => false,
        }
    }

    fn assemble_extension_candidates_for_trait(
        &mut self,
        import_ids: &SmallVec<[LocalDefId; 1]>,
        trait_def_id: DefId,
    ) -> Result<(), MethodError<'tcx>> {
        debug!("assemble_extension_candidates_for_trait(trait_def_id={:?})", trait_def_id);
        let trait_substs = self.fresh_item_substs(trait_def_id);
        let trait_ref = ty::TraitRef::new(trait_def_id, trait_substs);

        if self.tcx.is_trait_alias(trait_def_id) {
            // For trait aliases, assume all super-traits are relevant.
            let bounds = iter::once(trait_ref.to_poly_trait_ref());
            self.elaborate_bounds(bounds, |this, new_trait_ref, item| {
                let new_trait_ref = this.erase_late_bound_regions(new_trait_ref);

                let (xform_self_ty, xform_ret_ty) =
                    this.xform_self_ty(&item, new_trait_ref.self_ty(), new_trait_ref.substs);
                this.push_candidate(
                    Candidate {
                        xform_self_ty,
                        xform_ret_ty,
                        item,
                        import_ids: import_ids.clone(),
                        kind: TraitCandidate(new_trait_ref),
                    },
                    false,
                );
            });
        } else {
            debug_assert!(self.tcx.is_trait(trait_def_id));
            for item in self.impl_or_trait_item(trait_def_id) {
                // Check whether `trait_def_id` defines a method with suitable name.
                if !self.has_applicable_self(&item) {
                    debug!("method has inapplicable self");
                    self.record_static_candidate(TraitSource(trait_def_id));
                    continue;
                }

                let (xform_self_ty, xform_ret_ty) =
                    self.xform_self_ty(&item, trait_ref.self_ty(), trait_substs);
                self.push_candidate(
                    Candidate {
                        xform_self_ty,
                        xform_ret_ty,
                        item,
                        import_ids: import_ids.clone(),
                        kind: TraitCandidate(trait_ref),
                    },
                    false,
                );
            }
        }
        Ok(())
    }

    fn candidate_method_names(&self) -> Vec<Ident> {
        let mut set = FxHashSet::default();
        let mut names: Vec<_> = self
            .inherent_candidates
            .iter()
            .chain(&self.extension_candidates)
            .filter(|candidate| {
                if let Some(return_ty) = self.return_type {
                    self.matches_return_type(&candidate.item, None, return_ty)
                } else {
                    true
                }
            })
            .map(|candidate| candidate.item.ident)
            .filter(|&name| set.insert(name))
            .collect();

        // Sort them by the name so we have a stable result.
        names.sort_by_cached_key(|n| n.as_str());
        names
    }

    ///////////////////////////////////////////////////////////////////////////
    // THE ACTUAL SEARCH

    fn pick(mut self) -> PickResult<'tcx> {
        assert!(self.method_name.is_some());

        if let Some(r) = self.pick_core() {
            return r;
        }

        debug!("pick: actual search failed, assemble diagnostics");

        let static_candidates = mem::take(&mut self.static_candidates);
        let private_candidate = self.private_candidate.take();
        let unsatisfied_predicates = mem::take(&mut self.unsatisfied_predicates);

        // things failed, so lets look at all traits, for diagnostic purposes now:
        self.reset();

        let span = self.span;
        let tcx = self.tcx;

        self.assemble_extension_candidates_for_all_traits()?;

        let out_of_scope_traits = match self.pick_core() {
            Some(Ok(p)) => vec![p.item.container.id()],
            //Some(Ok(p)) => p.iter().map(|p| p.item.container().id()).collect(),
            Some(Err(MethodError::Ambiguity(v))) => v
                .into_iter()
                .map(|source| match source {
                    TraitSource(id) => id,
                    ImplSource(impl_id) => match tcx.trait_id_of_impl(impl_id) {
                        Some(id) => id,
                        None => span_bug!(span, "found inherent method when looking at traits"),
                    },
                })
                .collect(),
            Some(Err(MethodError::NoMatch(NoMatchData {
                out_of_scope_traits: others, ..
            }))) => {
                assert!(others.is_empty());
                vec![]
            }
            _ => vec![],
        };

        if let Some((kind, def_id)) = private_candidate {
            return Err(MethodError::PrivateMatch(kind, def_id, out_of_scope_traits));
        }
        let lev_candidate = self.probe_for_lev_candidate()?;

        Err(MethodError::NoMatch(NoMatchData::new(
            static_candidates,
            unsatisfied_predicates,
            out_of_scope_traits,
            lev_candidate,
            self.mode,
        )))
    }

    fn pick_core(&mut self) -> Option<PickResult<'tcx>> {
        let steps = self.steps.clone();

        // find the first step that works
        steps
            .iter()
            .filter(|step| {
                debug!("pick_core: step={:?}", step);
                // skip types that are from a type error or that would require dereferencing
                // a raw pointer
                !step.self_ty.references_error() && !step.from_unsafe_deref
            })
            .flat_map(|step| {
                let InferOk { value: self_ty, obligations: _ } = self
                    .fcx
                    .probe_instantiate_query_response(
                        self.span,
                        &self.orig_steps_var_values,
                        &step.self_ty,
                    )
                    .unwrap_or_else(|_| {
                        span_bug!(self.span, "{:?} was applicable but now isn't?", step.self_ty)
                    });
                self.pick_by_value_method(step, self_ty).or_else(|| {
                    self.pick_autorefd_method(step, self_ty, hir::Mutability::Not)
                        .or_else(|| self.pick_autorefd_method(step, self_ty, hir::Mutability::Mut))
                })
            })
            .next()
    }

    fn pick_by_value_method(
        &mut self,
        step: &CandidateStep<'tcx>,
        self_ty: Ty<'tcx>,
    ) -> Option<PickResult<'tcx>> {
        //! For each type `T` in the step list, this attempts to find a
        //! method where the (transformed) self type is exactly `T`. We
        //! do however do one transformation on the adjustment: if we
        //! are passing a region pointer in, we will potentially
        //! *reborrow* it to a shorter lifetime. This allows us to
        //! transparently pass `&mut` pointers, in particular, without
        //! consuming them for their entire lifetime.

        if step.unsize {
            return None;
        }

        self.pick_method(self_ty).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;

                // Insert a `&*` or `&mut *` if this is a reference type:
                if let ty::Ref(_, _, mutbl) = *step.self_ty.value.value.kind() {
                    pick.autoderefs += 1;
                    pick.autoref = Some(mutbl);
                }

                pick
            })
        })
    }

    fn pick_autorefd_method(
        &mut self,
        step: &CandidateStep<'tcx>,
        self_ty: Ty<'tcx>,
        mutbl: hir::Mutability,
    ) -> Option<PickResult<'tcx>> {
        let tcx = self.tcx;

        // In general, during probing we erase regions.
        let region = tcx.lifetimes.re_erased;

        let autoref_ty = tcx.mk_ref(region, ty::TypeAndMut { ty: self_ty, mutbl });
        self.pick_method(autoref_ty).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;
                pick.autoref = Some(mutbl);
                pick.unsize = step.unsize.then_some(self_ty);
                pick
            })
        })
    }

    fn pick_method(&mut self, self_ty: Ty<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_method(self_ty={})", self.ty_to_string(self_ty));

        let mut possibly_unsatisfied_predicates = Vec::new();
        let mut unstable_candidates = Vec::new();

        for (kind, candidates) in
            &[("inherent", &self.inherent_candidates), ("extension", &self.extension_candidates)]
        {
            debug!("searching {} candidates", kind);
            let res = self.consider_candidates(
                self_ty,
                candidates.iter(),
                &mut possibly_unsatisfied_predicates,
                Some(&mut unstable_candidates),
            );
            if let Some(pick) = res {
                if !self.is_suggestion.0 && !unstable_candidates.is_empty() {
                    if let Ok(p) = &pick {
                        // Emit a lint if there are unstable candidates alongside the stable ones.
                        //
                        // We suppress warning if we're picking the method only because it is a
                        // suggestion.
                        self.emit_unstable_name_collision_hint(p, &unstable_candidates);
                    }
                }
                return Some(pick);
            }
        }

        debug!("searching unstable candidates");
        let res = self.consider_candidates(
            self_ty,
            unstable_candidates.into_iter().map(|(c, _)| c),
            &mut possibly_unsatisfied_predicates,
            None,
        );
        if res.is_none() {
            self.unsatisfied_predicates.extend(possibly_unsatisfied_predicates);
        }
        res
    }

    fn consider_candidates<'b, ProbesIter>(
        &self,
        self_ty: Ty<'tcx>,
        probes: ProbesIter,
        possibly_unsatisfied_predicates: &mut Vec<(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
        )>,
        unstable_candidates: Option<&mut Vec<(&'b Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>>
    where
        ProbesIter: Iterator<Item = &'b Candidate<'tcx>> + Clone,
    {
        let mut applicable_candidates: Vec<_> = probes
            .clone()
            .map(|probe| {
                (probe, self.consider_probe(self_ty, probe, possibly_unsatisfied_predicates))
            })
            .filter(|&(_, status)| status != ProbeResult::NoMatch)
            .collect();

        debug!("applicable_candidates: {:?}", applicable_candidates);

        if applicable_candidates.len() > 1 {
            if let Some(pick) = self.collapse_candidates_to_trait_pick(&applicable_candidates[..]) {
                return Some(Ok(pick));
            }
        }

        if let Some(uc) = unstable_candidates {
            applicable_candidates.retain(|&(p, _)| {
                if let stability::EvalResult::Deny { feature, .. } =
                    self.tcx.eval_stability(p.item.def_id, None, self.span)
                {
                    uc.push((p, feature));
                    return false;
                }
                true
            });
        }

        if applicable_candidates.len() > 1 {
            let sources = probes.map(|p| self.candidate_source(p, self_ty)).collect();
            return Some(Err(MethodError::Ambiguity(sources)));
        }

        applicable_candidates.pop().map(|(probe, status)| {
            if status == ProbeResult::Match {
                Ok(probe.to_unadjusted_pick())
            } else {
                Err(MethodError::BadReturnType)
            }
        })
    }

    fn emit_unstable_name_collision_hint(
        &self,
        stable_pick: &Pick<'_>,
        unstable_candidates: &[(&Candidate<'tcx>, Symbol)],
    ) {
        self.tcx.struct_span_lint_hir(
            lint::builtin::UNSTABLE_NAME_COLLISIONS,
            self.fcx.body_id,
            self.span,
            |lint| {
                let mut diag = lint.build(
                    "a method with this name may be added to the standard library in the future",
                );
                // FIXME: This should be a `span_suggestion` instead of `help`
                // However `self.span` only
                // highlights the method name, so we can't use it. Also consider reusing the code from
                // `report_method_error()`.
                diag.help(&format!(
                    "call with fully qualified syntax `{}(...)` to keep using the current method",
                    self.tcx.def_path_str(stable_pick.item.def_id),
                ));

                if self.tcx.sess.is_nightly_build() {
                    for (candidate, feature) in unstable_candidates {
                        diag.help(&format!(
                            "add `#![feature({})]` to the crate attributes to enable `{}`",
                            feature,
                            self.tcx.def_path_str(candidate.item.def_id),
                        ));
                    }
                }

                diag.emit();
            },
        );
    }

    fn select_trait_candidate(
        &self,
        trait_ref: ty::TraitRef<'tcx>,
    ) -> traits::SelectionResult<'tcx, traits::Selection<'tcx>> {
        let cause = traits::ObligationCause::misc(self.span, self.body_id);
        let predicate = trait_ref.to_poly_trait_ref().to_poly_trait_predicate();
        let obligation = traits::Obligation::new(cause, self.param_env, predicate);
        traits::SelectionContext::new(self).select(&obligation)
    }

    fn candidate_source(&self, candidate: &Candidate<'tcx>, self_ty: Ty<'tcx>) -> CandidateSource {
        match candidate.kind {
            InherentImplCandidate(..) => ImplSource(candidate.item.container.id()),
            ObjectCandidate | WhereClauseCandidate(_) => TraitSource(candidate.item.container.id()),
            TraitCandidate(trait_ref) => self.probe(|_| {
                let _ = self
                    .at(&ObligationCause::dummy(), self.param_env)
                    .sup(candidate.xform_self_ty, self_ty);
                match self.select_trait_candidate(trait_ref) {
                    Ok(Some(traits::ImplSource::UserDefined(ref impl_data))) => {
                        // If only a single impl matches, make the error message point
                        // to that impl.
                        ImplSource(impl_data.impl_def_id)
                    }
                    _ => TraitSource(candidate.item.container.id()),
                }
            }),
        }
    }

    fn consider_probe(
        &self,
        self_ty: Ty<'tcx>,
        probe: &Candidate<'tcx>,
        possibly_unsatisfied_predicates: &mut Vec<(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
        )>,
    ) -> ProbeResult {
        debug!("consider_probe: self_ty={:?} probe={:?}", self_ty, probe);

        self.probe(|_| {
            // First check that the self type can be related.
            let sub_obligations = match self
                .at(&ObligationCause::dummy(), self.param_env)
                .sup(probe.xform_self_ty, self_ty)
            {
                Ok(InferOk { obligations, value: () }) => obligations,
                Err(_) => {
                    debug!("--> cannot relate self-types");
                    return ProbeResult::NoMatch;
                }
            };

            let mut result = ProbeResult::Match;
            let selcx = &mut traits::SelectionContext::new(self);
            let cause = traits::ObligationCause::misc(self.span, self.body_id);

            // If so, impls may carry other conditions (e.g., where
            // clauses) that must be considered. Make sure that those
            // match as well (or at least may match, sometimes we
            // don't have enough information to fully evaluate).
            match probe.kind {
                InherentImplCandidate(ref substs, ref ref_obligations) => {
                    // Check whether the impl imposes obligations we have to worry about.
                    let impl_def_id = probe.item.container.id();
                    let impl_bounds = self.tcx.predicates_of(impl_def_id);
                    let impl_bounds = impl_bounds.instantiate(self.tcx, substs);
                    let traits::Normalized { value: impl_bounds, obligations: norm_obligations } =
                        traits::normalize(selcx, self.param_env, cause.clone(), impl_bounds);

                    // Convert the bounds into obligations.
                    let impl_obligations =
                        traits::predicates_for_generics(cause, self.param_env, impl_bounds);

                    let candidate_obligations = impl_obligations
                        .chain(norm_obligations.into_iter())
                        .chain(ref_obligations.iter().cloned());
                    // Evaluate those obligations to see if they might possibly hold.
                    for o in candidate_obligations {
                        let o = self.resolve_vars_if_possible(o);
                        if !self.predicate_may_hold(&o) {
                            result = ProbeResult::NoMatch;
                            possibly_unsatisfied_predicates.push((o.predicate, None));
                        }
                    }
                }

                ObjectCandidate | WhereClauseCandidate(..) => {
                    // These have no additional conditions to check.
                }

                TraitCandidate(trait_ref) => {
                    let predicate = trait_ref.without_const().to_predicate(self.tcx);
                    let obligation = traits::Obligation::new(cause, self.param_env, predicate);
                    if !self.predicate_may_hold(&obligation) {
                        result = ProbeResult::NoMatch;
                        if self.probe(|_| {
                            match self.select_trait_candidate(trait_ref) {
                                Err(_) => return true,
                                Ok(Some(impl_source))
                                    if !impl_source.borrow_nested_obligations().is_empty() =>
                                {
                                    for obligation in impl_source.borrow_nested_obligations() {
                                        // Determine exactly which obligation wasn't met, so
                                        // that we can give more context in the error.
                                        if !self.predicate_may_hold(obligation) {
                                            let nested_predicate =
                                                self.resolve_vars_if_possible(obligation.predicate);
                                            let predicate =
                                                self.resolve_vars_if_possible(predicate);
                                            let p = if predicate == nested_predicate {
                                                // Avoid "`MyStruct: Foo` which is required by
                                                // `MyStruct: Foo`" in E0599.
                                                None
                                            } else {
                                                Some(predicate)
                                            };
                                            possibly_unsatisfied_predicates
                                                .push((nested_predicate, p));
                                        }
                                    }
                                }
                                _ => {
                                    // Some nested subobligation of this predicate
                                    // failed.
                                    let predicate = self.resolve_vars_if_possible(predicate);
                                    possibly_unsatisfied_predicates.push((predicate, None));
                                }
                            }
                            false
                        }) {
                            // This candidate's primary obligation doesn't even
                            // select - don't bother registering anything in
                            // `potentially_unsatisfied_predicates`.
                            return ProbeResult::NoMatch;
                        }
                    }
                }
            }

            // Evaluate those obligations to see if they might possibly hold.
            for o in sub_obligations {
                let o = self.resolve_vars_if_possible(o);
                if !self.predicate_may_hold(&o) {
                    result = ProbeResult::NoMatch;
                    possibly_unsatisfied_predicates.push((o.predicate, None));
                }
            }

            if let ProbeResult::Match = result {
                if let (Some(return_ty), Some(xform_ret_ty)) =
                    (self.return_type, probe.xform_ret_ty)
                {
                    let xform_ret_ty = self.resolve_vars_if_possible(xform_ret_ty);
                    debug!(
                        "comparing return_ty {:?} with xform ret ty {:?}",
                        return_ty, probe.xform_ret_ty
                    );
                    if self
                        .at(&ObligationCause::dummy(), self.param_env)
                        .sup(return_ty, xform_ret_ty)
                        .is_err()
                    {
                        return ProbeResult::BadReturnType;
                    }
                }
            }

            result
        })
    }

    /// Sometimes we get in a situation where we have multiple probes that are all impls of the
    /// same trait, but we don't know which impl to use. In this case, since in all cases the
    /// external interface of the method can be determined from the trait, it's ok not to decide.
    /// We can basically just collapse all of the probes for various impls into one where-clause
    /// probe. This will result in a pending obligation so when more type-info is available we can
    /// make the final decision.
    ///
    /// Example (`src/test/ui/method-two-trait-defer-resolution-1.rs`):
    ///
    /// ```
    /// trait Foo { ... }
    /// impl Foo for Vec<i32> { ... }
    /// impl Foo for Vec<usize> { ... }
    /// ```
    ///
    /// Now imagine the receiver is `Vec<_>`. It doesn't really matter at this time which impl we
    /// use, so it's ok to just commit to "using the method from the trait Foo".
    fn collapse_candidates_to_trait_pick(
        &self,
        probes: &[(&Candidate<'tcx>, ProbeResult)],
    ) -> Option<Pick<'tcx>> {
        // Do all probes correspond to the same trait?
        let container = probes[0].0.item.container;
        if let ty::ImplContainer(_) = container {
            return None;
        }
        if probes[1..].iter().any(|&(p, _)| p.item.container != container) {
            return None;
        }

        // FIXME: check the return type here somehow.
        // If so, just use this trait and call it a day.
        Some(Pick {
            item: probes[0].0.item,
            kind: TraitPick,
            import_ids: probes[0].0.import_ids.clone(),
            autoderefs: 0,
            autoref: None,
            unsize: None,
        })
    }

    /// Similarly to `probe_for_return_type`, this method attempts to find the best matching
    /// candidate method where the method name may have been misspelt. Similarly to other
    /// Levenshtein based suggestions, we provide at most one such suggestion.
    fn probe_for_lev_candidate(&mut self) -> Result<Option<ty::AssocItem>, MethodError<'tcx>> {
        debug!("probing for method names similar to {:?}", self.method_name);

        let steps = self.steps.clone();
        self.probe(|_| {
            let mut pcx = ProbeContext::new(
                self.fcx,
                self.span,
                self.mode,
                self.method_name,
                self.return_type,
                self.orig_steps_var_values.clone(),
                steps,
                IsSuggestion(true),
            );
            pcx.allow_similar_names = true;
            pcx.assemble_inherent_candidates();

            let method_names = pcx.candidate_method_names();
            pcx.allow_similar_names = false;
            let applicable_close_candidates: Vec<ty::AssocItem> = method_names
                .iter()
                .filter_map(|&method_name| {
                    pcx.reset();
                    pcx.method_name = Some(method_name);
                    pcx.assemble_inherent_candidates();
                    pcx.pick_core().and_then(|pick| pick.ok()).map(|pick| pick.item)
                })
                .collect();

            if applicable_close_candidates.is_empty() {
                Ok(None)
            } else {
                let best_name = {
                    let names = applicable_close_candidates
                        .iter()
                        .map(|cand| cand.ident.name)
                        .collect::<Vec<Symbol>>();
                    find_best_match_for_name(&names, self.method_name.unwrap().name, None)
                }
                .unwrap();
                Ok(applicable_close_candidates
                    .into_iter()
                    .find(|method| method.ident.name == best_name))
            }
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY
    fn has_applicable_self(&self, item: &ty::AssocItem) -> bool {
        // "Fast track" -- check for usage of sugar when in method call
        // mode.
        //
        // In Path mode (i.e., resolving a value like `T::next`), consider any
        // associated value (i.e., methods, constants) but not types.
        match self.mode {
            Mode::MethodCall => item.fn_has_self_parameter,
            Mode::Path => match item.kind {
                ty::AssocKind::Type => false,
                ty::AssocKind::Fn | ty::AssocKind::Const => true,
            },
        }
        // FIXME -- check for types that deref to `Self`,
        // like `Rc<Self>` and so on.
        //
        // Note also that the current code will break if this type
        // includes any of the type parameters defined on the method
        // -- but this could be overcome.
    }

    fn record_static_candidate(&mut self, source: CandidateSource) {
        self.static_candidates.push(source);
    }

    fn xform_self_ty(
        &self,
        item: &ty::AssocItem,
        impl_ty: Ty<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> (Ty<'tcx>, Option<Ty<'tcx>>) {
        if item.kind == ty::AssocKind::Fn && self.mode == Mode::MethodCall {
            let sig = self.xform_method_sig(item.def_id, substs);
            (sig.inputs()[0], Some(sig.output()))
        } else {
            (impl_ty, None)
        }
    }

    fn xform_method_sig(&self, method: DefId, substs: SubstsRef<'tcx>) -> ty::FnSig<'tcx> {
        let fn_sig = self.tcx.fn_sig(method);
        debug!("xform_self_ty(fn_sig={:?}, substs={:?})", fn_sig, substs);

        assert!(!substs.has_escaping_bound_vars());

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The substitutions we
        // are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        let generics = self.tcx.generics_of(method);
        assert_eq!(substs.len(), generics.parent_count as usize);

        // Erase any late-bound regions from the method and substitute
        // in the values from the substitution.
        let xform_fn_sig = self.erase_late_bound_regions(fn_sig);

        if generics.params.is_empty() {
            xform_fn_sig.subst(self.tcx, substs)
        } else {
            let substs = InternalSubsts::for_item(self.tcx, method, |param, _| {
                let i = param.index as usize;
                if i < substs.len() {
                    substs[i]
                } else {
                    match param.kind {
                        GenericParamDefKind::Lifetime => {
                            // In general, during probe we erase regions.
                            self.tcx.lifetimes.re_erased.into()
                        }
                        GenericParamDefKind::Type { .. } | GenericParamDefKind::Const => {
                            self.var_for_def(self.span, param)
                        }
                    }
                }
            });
            xform_fn_sig.subst(self.tcx, substs)
        }
    }

    /// Gets the type of an impl and generate substitutions with placeholders.
    fn impl_ty_and_substs(&self, impl_def_id: DefId) -> (Ty<'tcx>, SubstsRef<'tcx>) {
        (self.tcx.type_of(impl_def_id), self.fresh_item_substs(impl_def_id))
    }

    fn fresh_item_substs(&self, def_id: DefId) -> SubstsRef<'tcx> {
        InternalSubsts::for_item(self.tcx, def_id, |param, _| match param.kind {
            GenericParamDefKind::Lifetime => self.tcx.lifetimes.re_erased.into(),
            GenericParamDefKind::Type { .. } => self
                .next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::SubstitutionPlaceholder,
                    span: self.tcx.def_span(def_id),
                })
                .into(),
            GenericParamDefKind::Const { .. } => {
                let span = self.tcx.def_span(def_id);
                let origin = ConstVariableOrigin {
                    kind: ConstVariableOriginKind::SubstitutionPlaceholder,
                    span,
                };
                self.next_const_var(self.tcx.type_of(param.def_id), origin).into()
            }
        })
    }

    /// Replaces late-bound-regions bound by `value` with `'static` using
    /// `ty::erase_late_bound_regions`.
    ///
    /// This is only a reasonable thing to do during the *probe* phase, not the *confirm* phase, of
    /// method matching. It is reasonable during the probe phase because we don't consider region
    /// relationships at all. Therefore, we can just replace all the region variables with 'static
    /// rather than creating fresh region variables. This is nice for two reasons:
    ///
    /// 1. Because the numbers of the region variables would otherwise be fairly unique to this
    ///    particular method call, it winds up creating fewer types overall, which helps for memory
    ///    usage. (Admittedly, this is a rather small effect, though measurable.)
    ///
    /// 2. It makes it easier to deal with higher-ranked trait bounds, because we can replace any
    ///    late-bound regions with 'static. Otherwise, if we were going to replace late-bound
    ///    regions with actual region variables as is proper, we'd have to ensure that the same
    ///    region got replaced with the same variable, which requires a bit more coordination
    ///    and/or tracking the substitution and
    ///    so forth.
    fn erase_late_bound_regions<T>(&self, value: ty::Binder<T>) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.tcx.erase_late_bound_regions(value)
    }

    /// Finds the method with the appropriate name (or return type, as the case may be). If
    /// `allow_similar_names` is set, find methods with close-matching names.
    fn impl_or_trait_item(&self, def_id: DefId) -> Vec<ty::AssocItem> {
        if let Some(name) = self.method_name {
            if self.allow_similar_names {
                let max_dist = max(name.as_str().len(), 3) / 3;
                self.tcx
                    .associated_items(def_id)
                    .in_definition_order()
                    .filter(|x| {
                        let dist = lev_distance(&*name.as_str(), &x.ident.as_str());
                        x.kind.namespace() == Namespace::ValueNS && dist > 0 && dist <= max_dist
                    })
                    .copied()
                    .collect()
            } else {
                self.fcx
                    .associated_item(def_id, name, Namespace::ValueNS)
                    .map_or(Vec::new(), |x| vec![x])
            }
        } else {
            self.tcx.associated_items(def_id).in_definition_order().copied().collect()
        }
    }
}

impl<'tcx> Candidate<'tcx> {
    fn to_unadjusted_pick(&self) -> Pick<'tcx> {
        Pick {
            item: self.item,
            kind: match self.kind {
                InherentImplCandidate(..) => InherentImplPick,
                ObjectCandidate => ObjectPick,
                TraitCandidate(_) => TraitPick,
                WhereClauseCandidate(ref trait_ref) => {
                    // Only trait derived from where-clauses should
                    // appear here, so they should not contain any
                    // inference variables or other artifacts. This
                    // means they are safe to put into the
                    // `WhereClausePick`.
                    assert!(
                        !trait_ref.skip_binder().substs.needs_infer()
                            && !trait_ref.skip_binder().substs.has_placeholders()
                    );

                    WhereClausePick(*trait_ref)
                }
            },
            import_ids: self.import_ids.clone(),
            autoderefs: 0,
            autoref: None,
            unsize: None,
        }
    }
}
