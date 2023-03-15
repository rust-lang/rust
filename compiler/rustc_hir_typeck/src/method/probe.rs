use super::suggest;
use super::CandidateSource;
use super::MethodError;
use super::NoMatchData;

use crate::errors::MethodCallOnUnknownType;
use crate::FnCtxt;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir_analysis::astconv::InferCtxtExt as _;
use rustc_hir_analysis::autoderef::{self, Autoderef};
use rustc_infer::infer::canonical::OriginalQueryValues;
use rustc_infer::infer::canonical::{Canonical, QueryResponse};
use rustc_infer::infer::{self, InferOk, TyCtxtInferExt};
use rustc_middle::middle::stability;
use rustc_middle::ty::fast_reject::TreatProjections;
use rustc_middle::ty::fast_reject::{simplify_type, TreatParams};
use rustc_middle::ty::AssocItem;
use rustc_middle::ty::GenericParamDefKind;
use rustc_middle::ty::ToPredicate;
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_middle::ty::{InternalSubsts, SubstsRef};
use rustc_session::lint;
use rustc_span::def_id::DefId;
use rustc_span::def_id::LocalDefId;
use rustc_span::edit_distance::{
    edit_distance_with_substrings, find_best_match_for_name_with_substrings,
};
use rustc_span::symbol::sym;
use rustc_span::{symbol::Ident, Span, Symbol, DUMMY_SP};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::query::method_autoderef::MethodAutoderefBadTy;
use rustc_trait_selection::traits::query::method_autoderef::{
    CandidateStep, MethodAutoderefStepsResult,
};
use rustc_trait_selection::traits::query::CanonicalTyGoal;
use rustc_trait_selection::traits::NormalizeExt;
use rustc_trait_selection::traits::{self, ObligationCause};
use std::cell::RefCell;
use std::cmp::max;
use std::iter;
use std::ops::Deref;

use smallvec::{smallvec, SmallVec};

use self::CandidateKind::*;
pub use self::PickKind::*;

/// Boolean flag used to indicate if this search is for a suggestion
/// or not. If true, we can allow ambiguity and so forth.
#[derive(Clone, Copy, Debug)]
pub struct IsSuggestion(pub bool);

struct ProbeContext<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    mode: Mode,
    method_name: Option<Ident>,
    return_type: Option<Ty<'tcx>>,

    /// This is the OriginalQueryValues for the steps queries
    /// that are answered in steps.
    orig_steps_var_values: &'a OriginalQueryValues<'tcx>,
    steps: &'tcx [CandidateStep<'tcx>],

    inherent_candidates: Vec<Candidate<'tcx>>,
    extension_candidates: Vec<Candidate<'tcx>>,
    impl_dups: FxHashSet<DefId>,

    /// When probing for names, include names that are close to the
    /// requested name (by edit distance)
    allow_similar_names: bool,

    /// Some(candidate) if there is a private candidate
    private_candidate: Option<(DefKind, DefId)>,

    /// Collects near misses when the candidate functions are missing a `self` keyword and is only
    /// used for error reporting
    static_candidates: RefCell<Vec<CandidateSource>>,

    /// Collects near misses when trait bounds for type parameters are unsatisfied and is only used
    /// for error reporting
    unsatisfied_predicates: RefCell<
        Vec<(ty::Predicate<'tcx>, Option<ty::Predicate<'tcx>>, Option<ObligationCause<'tcx>>)>,
    >,

    scope_expr_id: hir::HirId,
}

impl<'a, 'tcx> Deref for ProbeContext<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        self.fcx
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Candidate<'tcx> {
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
    pub(crate) item: ty::AssocItem,
    pub(crate) kind: CandidateKind<'tcx>,
    pub(crate) import_ids: SmallVec<[LocalDefId; 1]>,
}

#[derive(Debug, Clone)]
pub(crate) enum CandidateKind<'tcx> {
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

/// When adjusting a receiver we often want to do one of
///
/// - Add a `&` (or `&mut`), converting the receiver from `T` to `&T` (or `&mut T`)
/// - If the receiver has type `*mut T`, convert it to `*const T`
///
/// This type tells us which one to do.
///
/// Note that in principle we could do both at the same time. For example, when the receiver has
/// type `T`, we could autoref it to `&T`, then convert to `*const T`. Or, when it has type `*mut
/// T`, we could convert it to `*const T`, then autoref to `&*const T`. However, currently we do
/// (at most) one of these. Either the receiver has type `T` and we convert it to `&T` (or with
/// `mut`), or it has type `*mut T` and we convert it to `*const T`.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum AutorefOrPtrAdjustment {
    /// Receiver has type `T`, add `&` or `&mut` (it `T` is `mut`), and maybe also "unsize" it.
    /// Unsizing is used to convert a `[T; N]` to `[T]`, which only makes sense when autorefing.
    Autoref {
        mutbl: hir::Mutability,

        /// Indicates that the source expression should be "unsized" to a target type.
        /// This is special-cased for just arrays unsizing to slices.
        unsize: bool,
    },
    /// Receiver has type `*mut T`, convert to `*const T`
    ToConstPtr,
}

impl AutorefOrPtrAdjustment {
    fn get_unsize(&self) -> bool {
        match self {
            AutorefOrPtrAdjustment::Autoref { mutbl: _, unsize } => *unsize,
            AutorefOrPtrAdjustment::ToConstPtr => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pick<'tcx> {
    pub item: ty::AssocItem,
    pub kind: PickKind<'tcx>,
    pub import_ids: SmallVec<[LocalDefId; 1]>,

    /// Indicates that the source expression should be autoderef'd N times
    /// ```ignore (not-rust)
    /// A = expr | *expr | **expr | ...
    /// ```
    pub autoderefs: usize,

    /// Indicates that we want to add an autoref (and maybe also unsize it), or if the receiver is
    /// `*mut T`, convert it to `*const T`.
    pub autoref_or_ptr_adjustment: Option<AutorefOrPtrAdjustment>,
    pub self_ty: Ty<'tcx>,

    /// Unstable candidates alongside the stable ones.
    unstable_candidates: Vec<(Candidate<'tcx>, Symbol)>,
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
    // `self` argument of the method, and static methods aren't considered.
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
    #[instrument(level = "debug", skip(self, candidate_filter))]
    pub fn probe_for_return_type(
        &self,
        span: Span,
        mode: Mode,
        return_type: Ty<'tcx>,
        self_ty: Ty<'tcx>,
        scope_expr_id: hir::HirId,
        candidate_filter: impl Fn(&ty::AssocItem) -> bool,
    ) -> Vec<ty::AssocItem> {
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
                |probe_cx| Ok(probe_cx.candidate_method_names(candidate_filter)),
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

    #[instrument(level = "debug", skip(self))]
    pub fn probe_for_name(
        &self,
        mode: Mode,
        item_name: Ident,
        return_type: Option<Ty<'tcx>>,
        is_suggestion: IsSuggestion,
        self_ty: Ty<'tcx>,
        scope_expr_id: hir::HirId,
        scope: ProbeScope,
    ) -> PickResult<'tcx> {
        self.probe_op(
            item_name.span,
            mode,
            Some(item_name),
            return_type,
            is_suggestion,
            self_ty,
            scope_expr_id,
            scope,
            |probe_cx| probe_cx.pick(),
        )
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn probe_for_name_many(
        &self,
        mode: Mode,
        item_name: Ident,
        return_type: Option<Ty<'tcx>>,
        is_suggestion: IsSuggestion,
        self_ty: Ty<'tcx>,
        scope_expr_id: hir::HirId,
        scope: ProbeScope,
    ) -> Vec<Candidate<'tcx>> {
        self.probe_op(
            item_name.span,
            mode,
            Some(item_name),
            return_type,
            is_suggestion,
            self_ty,
            scope_expr_id,
            scope,
            |probe_cx| {
                Ok(probe_cx
                    .inherent_candidates
                    .into_iter()
                    .chain(probe_cx.extension_candidates)
                    .collect())
            },
        )
        .unwrap()
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
        OP: FnOnce(ProbeContext<'_, 'tcx>) -> Result<R, MethodError<'tcx>>,
    {
        let mut orig_values = OriginalQueryValues::default();
        let param_env_and_self_ty = self.canonicalize_query(
            ParamEnvAnd { param_env: self.param_env, value: self_ty },
            &mut orig_values,
        );

        let steps = match mode {
            Mode::MethodCall => self.tcx.method_autoderef_steps(param_env_and_self_ty),
            Mode::Path => self.probe(|_| {
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
                    steps: infcx.tcx.arena.alloc_from_iter([CandidateStep {
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
            }),
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
                        "type annotations needed",
                        |lint| lint,
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
                return Err(MethodError::NoMatch(NoMatchData {
                    static_candidates: Vec::new(),
                    unsatisfied_predicates: Vec::new(),
                    out_of_scope_traits: Vec::new(),
                    similar_candidate: None,
                    mode,
                }));
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
                &orig_values,
                steps.steps,
                scope_expr_id,
            );

            probe_cx.assemble_inherent_candidates();
            match scope {
                ProbeScope::TraitsInScope => {
                    probe_cx.assemble_extension_candidates_for_traits_in_scope()
                }
                ProbeScope::AllTraits => probe_cx.assemble_extension_candidates_for_all_traits(),
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

    let (ref infcx, goal, inference_vars) = tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &goal);
    let ParamEnvAnd { param_env, value: self_ty } = goal;

    let mut autoderef =
        Autoderef::new(infcx, param_env, hir::def_id::CRATE_DEF_ID, DUMMY_SP, self_ty)
            .include_raw_pointers()
            .silence_errors();
    let mut reached_raw_pointer = false;
    let mut steps: Vec<_> = autoderef
        .by_ref()
        .map(|(ty, d)| {
            let step = CandidateStep {
                self_ty: infcx.make_query_response_ignoring_pending_obligations(inference_vars, ty),
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
            ty: infcx.make_query_response_ignoring_pending_obligations(inference_vars, final_ty),
        }),
        ty::Array(elem_ty, _) => {
            let dereferences = steps.len() - 1;

            steps.push(CandidateStep {
                self_ty: infcx.make_query_response_ignoring_pending_obligations(
                    inference_vars,
                    infcx.tcx.mk_slice(*elem_ty),
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
        steps: tcx.arena.alloc_from_iter(steps),
        opt_bad_ty: opt_bad_ty.map(|ty| &*tcx.arena.alloc(ty)),
        reached_recursion_limit: autoderef.reached_recursion_limit(),
    }
}

impl<'a, 'tcx> ProbeContext<'a, 'tcx> {
    fn new(
        fcx: &'a FnCtxt<'a, 'tcx>,
        span: Span,
        mode: Mode,
        method_name: Option<Ident>,
        return_type: Option<Ty<'tcx>>,
        orig_steps_var_values: &'a OriginalQueryValues<'tcx>,
        steps: &'tcx [CandidateStep<'tcx>],
        scope_expr_id: hir::HirId,
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
            allow_similar_names: false,
            private_candidate: None,
            static_candidates: RefCell::new(Vec::new()),
            unsatisfied_predicates: RefCell::new(Vec::new()),
            scope_expr_id,
        }
    }

    fn reset(&mut self) {
        self.inherent_candidates.clear();
        self.extension_candidates.clear();
        self.impl_dups.clear();
        self.private_candidate = None;
        self.static_candidates.borrow_mut().clear();
        self.unsatisfied_predicates.borrow_mut().clear();
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY

    fn push_candidate(&mut self, candidate: Candidate<'tcx>, is_inherent: bool) {
        let is_accessible = if let Some(name) = self.method_name {
            let item = candidate.item;
            let hir_id = self.tcx.hir().local_def_id_to_hir_id(self.body_id);
            let def_scope =
                self.tcx.adjust_ident_and_get_scope(name, item.container_id(self.tcx), hir_id).1;
            item.visibility(self.tcx).is_accessible_from(def_scope, self.tcx)
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
        for step in self.steps.iter() {
            self.assemble_probe(&step.self_ty);
        }
    }

    fn assemble_probe(&mut self, self_ty: &Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>) {
        debug!("assemble_probe: self_ty={:?}", self_ty);
        let raw_self_ty = self_ty.value.value;
        match *raw_self_ty.kind() {
            ty::Dynamic(data, ..) if let Some(p) = data.principal() => {
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
                        .instantiate_canonical_with_fresh_inference_vars(self.span, self_ty);

                self.assemble_inherent_candidates_from_object(generalized_self_ty);
                self.assemble_inherent_impl_candidates_for_type(p.def_id());
                if self.tcx.has_attr(p.def_id(), sym::rustc_has_incoherent_inherent_impls) {
                    self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty);
                }
            }
            ty::Adt(def, _) => {
                let def_id = def.did();
                self.assemble_inherent_impl_candidates_for_type(def_id);
                if self.tcx.has_attr(def_id, sym::rustc_has_incoherent_inherent_impls) {
                    self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty);
                }
            }
            ty::Foreign(did) => {
                self.assemble_inherent_impl_candidates_for_type(did);
                if self.tcx.has_attr(did, sym::rustc_has_incoherent_inherent_impls) {
                    self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty);
                }
            }
            ty::Param(p) => {
                self.assemble_inherent_candidates_from_param(p);
            }
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(..)
            | ty::Never
            | ty::Tuple(..) => self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty),
            _ => {}
        }
    }

    fn assemble_inherent_candidates_for_incoherent_ty(&mut self, self_ty: Ty<'tcx>) {
        let Some(simp) = simplify_type(self.tcx, self_ty, TreatParams::AsCandidateKey, TreatProjections::AsCandidateKey) else {
            bug!("unexpected incoherent type: {:?}", self_ty)
        };
        for &impl_def_id in self.tcx.incoherent_impls(simp) {
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
                self.record_static_candidate(CandidateSource::Impl(impl_def_id));
                continue;
            }

            let (impl_ty, impl_substs) = self.impl_ty_and_substs(impl_def_id);
            let impl_ty = impl_ty.subst(self.tcx, impl_substs);

            debug!("impl_ty: {:?}", impl_ty);

            // Determine the receiver type that the method itself expects.
            let (xform_self_ty, xform_ret_ty) = self.xform_self_ty(item, impl_ty, impl_substs);
            debug!("xform_self_ty: {:?}, xform_ret_ty: {:?}", xform_self_ty, xform_ret_ty);

            // We can't use normalize_associated_types_in as it will pollute the
            // fcx's fulfillment context after this probe is over.
            // Note: we only normalize `xform_self_ty` here since the normalization
            // of the return type can lead to inference results that prohibit
            // valid candidates from being found, see issue #85671
            // FIXME Postponing the normalization of the return type likely only hides a deeper bug,
            // which might be caused by the `param_env` itself. The clauses of the `param_env`
            // maybe shouldn't include `Param`s, but rather fresh variables or be canonicalized,
            // see issue #89650
            let cause = traits::ObligationCause::misc(self.span, self.body_id);
            let InferOk { value: xform_self_ty, obligations } =
                self.fcx.at(&cause, self.param_env).normalize(xform_self_ty);

            debug!(
                "assemble_inherent_impl_probe after normalization: xform_self_ty = {:?}/{:?}",
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

        // It is illegal to invoke a method on a trait instance that refers to
        // the `Self` type. An [`ObjectSafetyViolation::SupertraitSelf`] error
        // will be reported by `object_safety.rs` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use a
        // substitution that replaces `Self` with the object type itself. Hence,
        // a `&self` method will wind up with an argument type like `&dyn Trait`.
        let trait_ref = principal.with_self_ty(self.tcx, self_ty);
        self.elaborate_bounds(iter::once(trait_ref), |this, new_trait_ref, item| {
            let new_trait_ref = this.erase_late_bound_regions(new_trait_ref);

            let (xform_self_ty, xform_ret_ty) =
                this.xform_self_ty(item, new_trait_ref.self_ty(), new_trait_ref.substs);
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
            let bound_predicate = predicate.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::Clause::Trait(trait_predicate)) => {
                    match *trait_predicate.trait_ref.self_ty().kind() {
                        ty::Param(p) if p == param_ty => {
                            Some(bound_predicate.rebind(trait_predicate.trait_ref))
                        }
                        _ => None,
                    }
                }
                ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..))
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::Clause::Projection(..))
                | ty::PredicateKind::Clause(ty::Clause::RegionOutlives(..))
                | ty::PredicateKind::WellFormed(..)
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::Clause(ty::Clause::TypeOutlives(..))
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::AliasEq(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
            }
        });

        self.elaborate_bounds(bounds, |this, poly_trait_ref, item| {
            let trait_ref = this.erase_late_bound_regions(poly_trait_ref);

            let (xform_self_ty, xform_ret_ty) =
                this.xform_self_ty(item, trait_ref.self_ty(), trait_ref.substs);

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
                    self.record_static_candidate(CandidateSource::Trait(bound_trait_ref.def_id()));
                } else {
                    mk_cand(self, bound_trait_ref, item);
                }
            }
        }
    }

    fn assemble_extension_candidates_for_traits_in_scope(&mut self) {
        let mut duplicates = FxHashSet::default();
        let opt_applicable_traits = self.tcx.in_scope_traits(self.scope_expr_id);
        if let Some(applicable_traits) = opt_applicable_traits {
            for trait_candidate in applicable_traits.iter() {
                let trait_did = trait_candidate.def_id;
                if duplicates.insert(trait_did) {
                    self.assemble_extension_candidates_for_trait(
                        &trait_candidate.import_ids,
                        trait_did,
                    );
                }
            }
        }
    }

    fn assemble_extension_candidates_for_all_traits(&mut self) {
        let mut duplicates = FxHashSet::default();
        for trait_info in suggest::all_traits(self.tcx) {
            if duplicates.insert(trait_info.def_id) {
                self.assemble_extension_candidates_for_trait(&smallvec![], trait_info.def_id);
            }
        }
    }

    fn matches_return_type(
        &self,
        method: ty::AssocItem,
        self_ty: Option<Ty<'tcx>>,
        expected: Ty<'tcx>,
    ) -> bool {
        match method.kind {
            ty::AssocKind::Fn => self.probe(|_| {
                let substs = self.fresh_substs_for_item(self.span, method.def_id);
                let fty = self.tcx.fn_sig(method.def_id).subst(self.tcx, substs);
                let fty = self.instantiate_binder_with_fresh_vars(self.span, infer::FnCall, fty);

                if let Some(self_ty) = self_ty {
                    if self
                        .at(&ObligationCause::dummy(), self.param_env)
                        .sup(fty.inputs()[0], self_ty)
                        .is_err()
                    {
                        return false;
                    }
                }
                self.can_sub(self.param_env, fty.output(), expected)
            }),
            _ => false,
        }
    }

    fn assemble_extension_candidates_for_trait(
        &mut self,
        import_ids: &SmallVec<[LocalDefId; 1]>,
        trait_def_id: DefId,
    ) {
        debug!("assemble_extension_candidates_for_trait(trait_def_id={:?})", trait_def_id);
        let trait_substs = self.fresh_item_substs(trait_def_id);
        let trait_ref = self.tcx.mk_trait_ref(trait_def_id, trait_substs);

        if self.tcx.is_trait_alias(trait_def_id) {
            // For trait aliases, recursively assume all explicitly named traits are relevant
            for expansion in traits::expand_trait_aliases(
                self.tcx,
                iter::once((ty::Binder::dummy(trait_ref), self.span)),
            ) {
                let bound_trait_ref = expansion.trait_ref();
                for item in self.impl_or_trait_item(bound_trait_ref.def_id()) {
                    if !self.has_applicable_self(&item) {
                        self.record_static_candidate(CandidateSource::Trait(
                            bound_trait_ref.def_id(),
                        ));
                    } else {
                        let new_trait_ref = self.erase_late_bound_regions(bound_trait_ref);

                        let (xform_self_ty, xform_ret_ty) =
                            self.xform_self_ty(item, new_trait_ref.self_ty(), new_trait_ref.substs);
                        self.push_candidate(
                            Candidate {
                                xform_self_ty,
                                xform_ret_ty,
                                item,
                                import_ids: import_ids.clone(),
                                kind: TraitCandidate(new_trait_ref),
                            },
                            false,
                        );
                    }
                }
            }
        } else {
            debug_assert!(self.tcx.is_trait(trait_def_id));
            if self.tcx.trait_is_auto(trait_def_id) {
                return;
            }
            for item in self.impl_or_trait_item(trait_def_id) {
                // Check whether `trait_def_id` defines a method with suitable name.
                if !self.has_applicable_self(&item) {
                    debug!("method has inapplicable self");
                    self.record_static_candidate(CandidateSource::Trait(trait_def_id));
                    continue;
                }

                let (xform_self_ty, xform_ret_ty) =
                    self.xform_self_ty(item, trait_ref.self_ty(), trait_substs);
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
    }

    fn candidate_method_names(
        &self,
        candidate_filter: impl Fn(&ty::AssocItem) -> bool,
    ) -> Vec<Ident> {
        let mut set = FxHashSet::default();
        let mut names: Vec<_> = self
            .inherent_candidates
            .iter()
            .chain(&self.extension_candidates)
            .filter(|candidate| candidate_filter(&candidate.item))
            .filter(|candidate| {
                if let Some(return_ty) = self.return_type {
                    self.matches_return_type(candidate.item, None, return_ty)
                } else {
                    true
                }
            })
            .map(|candidate| candidate.item.ident(self.tcx))
            .filter(|&name| set.insert(name))
            .collect();

        // Sort them by the name so we have a stable result.
        names.sort_by(|a, b| a.as_str().cmp(b.as_str()));
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

        let static_candidates = std::mem::take(self.static_candidates.get_mut());
        let private_candidate = self.private_candidate.take();
        let unsatisfied_predicates = std::mem::take(self.unsatisfied_predicates.get_mut());

        // things failed, so lets look at all traits, for diagnostic purposes now:
        self.reset();

        let span = self.span;
        let tcx = self.tcx;

        self.assemble_extension_candidates_for_all_traits();

        let out_of_scope_traits = match self.pick_core() {
            Some(Ok(p)) => vec![p.item.container_id(self.tcx)],
            Some(Err(MethodError::Ambiguity(v))) => v
                .into_iter()
                .map(|source| match source {
                    CandidateSource::Trait(id) => id,
                    CandidateSource::Impl(impl_id) => match tcx.trait_id_of_impl(impl_id) {
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
        let similar_candidate = self.probe_for_similar_candidate()?;

        Err(MethodError::NoMatch(NoMatchData {
            static_candidates,
            unsatisfied_predicates,
            out_of_scope_traits,
            similar_candidate,
            mode: self.mode,
        }))
    }

    fn pick_core(&self) -> Option<PickResult<'tcx>> {
        // Pick stable methods only first, and consider unstable candidates if not found.
        self.pick_all_method(Some(&mut vec![])).or_else(|| self.pick_all_method(None))
    }

    fn pick_all_method(
        &self,
        mut unstable_candidates: Option<&mut Vec<(Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>> {
        self.steps
            .iter()
            .filter(|step| {
                debug!("pick_all_method: step={:?}", step);
                // skip types that are from a type error or that would require dereferencing
                // a raw pointer
                !step.self_ty.references_error() && !step.from_unsafe_deref
            })
            .find_map(|step| {
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
                self.pick_by_value_method(step, self_ty, unstable_candidates.as_deref_mut())
                    .or_else(|| {
                        self.pick_autorefd_method(
                            step,
                            self_ty,
                            hir::Mutability::Not,
                            unstable_candidates.as_deref_mut(),
                        )
                        .or_else(|| {
                            self.pick_autorefd_method(
                                step,
                                self_ty,
                                hir::Mutability::Mut,
                                unstable_candidates.as_deref_mut(),
                            )
                        })
                        .or_else(|| {
                            self.pick_const_ptr_method(
                                step,
                                self_ty,
                                unstable_candidates.as_deref_mut(),
                            )
                        })
                    })
            })
    }

    /// For each type `T` in the step list, this attempts to find a method where
    /// the (transformed) self type is exactly `T`. We do however do one
    /// transformation on the adjustment: if we are passing a region pointer in,
    /// we will potentially *reborrow* it to a shorter lifetime. This allows us
    /// to transparently pass `&mut` pointers, in particular, without consuming
    /// them for their entire lifetime.
    fn pick_by_value_method(
        &self,
        step: &CandidateStep<'tcx>,
        self_ty: Ty<'tcx>,
        unstable_candidates: Option<&mut Vec<(Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>> {
        if step.unsize {
            return None;
        }

        self.pick_method(self_ty, unstable_candidates).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;

                // Insert a `&*` or `&mut *` if this is a reference type:
                if let ty::Ref(_, _, mutbl) = *step.self_ty.value.value.kind() {
                    pick.autoderefs += 1;
                    pick.autoref_or_ptr_adjustment = Some(AutorefOrPtrAdjustment::Autoref {
                        mutbl,
                        unsize: pick.autoref_or_ptr_adjustment.map_or(false, |a| a.get_unsize()),
                    })
                }

                pick
            })
        })
    }

    fn pick_autorefd_method(
        &self,
        step: &CandidateStep<'tcx>,
        self_ty: Ty<'tcx>,
        mutbl: hir::Mutability,
        unstable_candidates: Option<&mut Vec<(Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>> {
        let tcx = self.tcx;

        // In general, during probing we erase regions.
        let region = tcx.lifetimes.re_erased;

        let autoref_ty = tcx.mk_ref(region, ty::TypeAndMut { ty: self_ty, mutbl });
        self.pick_method(autoref_ty, unstable_candidates).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;
                pick.autoref_or_ptr_adjustment =
                    Some(AutorefOrPtrAdjustment::Autoref { mutbl, unsize: step.unsize });
                pick
            })
        })
    }

    /// If `self_ty` is `*mut T` then this picks `*const T` methods. The reason why we have a
    /// special case for this is because going from `*mut T` to `*const T` with autoderefs and
    /// autorefs would require dereferencing the pointer, which is not safe.
    fn pick_const_ptr_method(
        &self,
        step: &CandidateStep<'tcx>,
        self_ty: Ty<'tcx>,
        unstable_candidates: Option<&mut Vec<(Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>> {
        // Don't convert an unsized reference to ptr
        if step.unsize {
            return None;
        }

        let &ty::RawPtr(ty::TypeAndMut { ty, mutbl: hir::Mutability::Mut }) = self_ty.kind() else {
            return None;
        };

        let const_self_ty = ty::TypeAndMut { ty, mutbl: hir::Mutability::Not };
        let const_ptr_ty = self.tcx.mk_ptr(const_self_ty);
        self.pick_method(const_ptr_ty, unstable_candidates).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;
                pick.autoref_or_ptr_adjustment = Some(AutorefOrPtrAdjustment::ToConstPtr);
                pick
            })
        })
    }

    fn pick_method(
        &self,
        self_ty: Ty<'tcx>,
        mut unstable_candidates: Option<&mut Vec<(Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>> {
        debug!("pick_method(self_ty={})", self.ty_to_string(self_ty));

        let mut possibly_unsatisfied_predicates = Vec::new();

        for (kind, candidates) in
            &[("inherent", &self.inherent_candidates), ("extension", &self.extension_candidates)]
        {
            debug!("searching {} candidates", kind);
            let res = self.consider_candidates(
                self_ty,
                candidates,
                &mut possibly_unsatisfied_predicates,
                unstable_candidates.as_deref_mut(),
            );
            if let Some(pick) = res {
                return Some(pick);
            }
        }

        // `pick_method` may be called twice for the same self_ty if no stable methods
        // match. Only extend once.
        if unstable_candidates.is_some() {
            self.unsatisfied_predicates.borrow_mut().extend(possibly_unsatisfied_predicates);
        }
        None
    }

    fn consider_candidates(
        &self,
        self_ty: Ty<'tcx>,
        candidates: &[Candidate<'tcx>],
        possibly_unsatisfied_predicates: &mut Vec<(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )>,
        mut unstable_candidates: Option<&mut Vec<(Candidate<'tcx>, Symbol)>>,
    ) -> Option<PickResult<'tcx>> {
        let mut applicable_candidates: Vec<_> = candidates
            .iter()
            .map(|probe| {
                (probe, self.consider_probe(self_ty, probe, possibly_unsatisfied_predicates))
            })
            .filter(|&(_, status)| status != ProbeResult::NoMatch)
            .collect();

        debug!("applicable_candidates: {:?}", applicable_candidates);

        if applicable_candidates.len() > 1 {
            if let Some(pick) =
                self.collapse_candidates_to_trait_pick(self_ty, &applicable_candidates)
            {
                return Some(Ok(pick));
            }
        }

        if let Some(uc) = &mut unstable_candidates {
            applicable_candidates.retain(|&(candidate, _)| {
                if let stability::EvalResult::Deny { feature, .. } =
                    self.tcx.eval_stability(candidate.item.def_id, None, self.span, None)
                {
                    uc.push((candidate.clone(), feature));
                    return false;
                }
                true
            });
        }

        if applicable_candidates.len() > 1 {
            let sources = candidates.iter().map(|p| self.candidate_source(p, self_ty)).collect();
            return Some(Err(MethodError::Ambiguity(sources)));
        }

        applicable_candidates.pop().map(|(probe, status)| match status {
            ProbeResult::Match => {
                Ok(probe
                    .to_unadjusted_pick(self_ty, unstable_candidates.cloned().unwrap_or_default()))
            }
            ProbeResult::NoMatch | ProbeResult::BadReturnType => Err(MethodError::BadReturnType),
        })
    }
}

impl<'tcx> Pick<'tcx> {
    /// In case there were unstable name collisions, emit them as a lint.
    /// Checks whether two picks do not refer to the same trait item for the same `Self` type.
    /// Only useful for comparisons of picks in order to improve diagnostics.
    /// Do not use for type checking.
    pub fn differs_from(&self, other: &Self) -> bool {
        let Self {
            item:
                AssocItem {
                    def_id,
                    name: _,
                    kind: _,
                    container: _,
                    trait_item_def_id: _,
                    fn_has_self_parameter: _,
                    opt_rpitit_info: _,
                },
            kind: _,
            import_ids: _,
            autoderefs: _,
            autoref_or_ptr_adjustment: _,
            self_ty,
            unstable_candidates: _,
        } = *self;
        self_ty != other.self_ty || def_id != other.item.def_id
    }

    /// In case there were unstable name collisions, emit them as a lint.
    pub fn maybe_emit_unstable_name_collision_hint(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
        scope_expr_id: hir::HirId,
    ) {
        if self.unstable_candidates.is_empty() {
            return;
        }
        let def_kind = self.item.kind.as_def_kind();
        tcx.struct_span_lint_hir(
            lint::builtin::UNSTABLE_NAME_COLLISIONS,
            scope_expr_id,
            span,
            format!(
                "{} {} with this name may be added to the standard library in the future",
                tcx.def_kind_descr_article(def_kind, self.item.def_id),
                tcx.def_kind_descr(def_kind, self.item.def_id),
            ),
            |lint| {
                match (self.item.kind, self.item.container) {
                    (ty::AssocKind::Fn, _) => {
                        // FIXME: This should be a `span_suggestion` instead of `help`
                        // However `self.span` only
                        // highlights the method name, so we can't use it. Also consider reusing
                        // the code from `report_method_error()`.
                        lint.help(&format!(
                            "call with fully qualified syntax `{}(...)` to keep using the current \
                             method",
                            tcx.def_path_str(self.item.def_id),
                        ));
                    }
                    (ty::AssocKind::Const, ty::AssocItemContainer::TraitContainer) => {
                        let def_id = self.item.container_id(tcx);
                        lint.span_suggestion(
                            span,
                            "use the fully qualified path to the associated const",
                            format!(
                                "<{} as {}>::{}",
                                self.self_ty,
                                tcx.def_path_str(def_id),
                                self.item.name
                            ),
                            Applicability::MachineApplicable,
                        );
                    }
                    _ => {}
                }
                if tcx.sess.is_nightly_build() {
                    for (candidate, feature) in &self.unstable_candidates {
                        lint.help(&format!(
                            "add `#![feature({})]` to the crate attributes to enable `{}`",
                            feature,
                            tcx.def_path_str(candidate.item.def_id),
                        ));
                    }
                }

                lint
            },
        );
    }
}

impl<'a, 'tcx> ProbeContext<'a, 'tcx> {
    fn select_trait_candidate(
        &self,
        trait_ref: ty::TraitRef<'tcx>,
    ) -> traits::SelectionResult<'tcx, traits::Selection<'tcx>> {
        let cause = traits::ObligationCause::misc(self.span, self.body_id);
        let predicate = ty::Binder::dummy(trait_ref);
        let obligation = traits::Obligation::new(self.tcx, cause, self.param_env, predicate);
        traits::SelectionContext::new(self).select(&obligation)
    }

    fn candidate_source(&self, candidate: &Candidate<'tcx>, self_ty: Ty<'tcx>) -> CandidateSource {
        match candidate.kind {
            InherentImplCandidate(..) => {
                CandidateSource::Impl(candidate.item.container_id(self.tcx))
            }
            ObjectCandidate | WhereClauseCandidate(_) => {
                CandidateSource::Trait(candidate.item.container_id(self.tcx))
            }
            TraitCandidate(trait_ref) => self.probe(|_| {
                let _ = self
                    .at(&ObligationCause::dummy(), self.param_env)
                    .sup(candidate.xform_self_ty, self_ty);
                match self.select_trait_candidate(trait_ref) {
                    Ok(Some(traits::ImplSource::UserDefined(ref impl_data))) => {
                        // If only a single impl matches, make the error message point
                        // to that impl.
                        CandidateSource::Impl(impl_data.impl_def_id)
                    }
                    _ => CandidateSource::Trait(candidate.item.container_id(self.tcx)),
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
            Option<ObligationCause<'tcx>>,
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
                Err(err) => {
                    debug!("--> cannot relate self-types {:?}", err);
                    return ProbeResult::NoMatch;
                }
            };

            let mut result = ProbeResult::Match;
            let mut xform_ret_ty = probe.xform_ret_ty;
            debug!(?xform_ret_ty);

            let cause = traits::ObligationCause::misc(self.span, self.body_id);

            let mut parent_pred = None;

            // If so, impls may carry other conditions (e.g., where
            // clauses) that must be considered. Make sure that those
            // match as well (or at least may match, sometimes we
            // don't have enough information to fully evaluate).
            match probe.kind {
                InherentImplCandidate(ref substs, ref ref_obligations) => {
                    // `xform_ret_ty` hasn't been normalized yet, only `xform_self_ty`,
                    // see the reasons mentioned in the comments in `assemble_inherent_impl_probe`
                    // for why this is necessary
                    let InferOk {
                        value: normalized_xform_ret_ty,
                        obligations: normalization_obligations,
                    } = self.fcx.at(&cause, self.param_env).normalize(xform_ret_ty);
                    xform_ret_ty = normalized_xform_ret_ty;
                    debug!("xform_ret_ty after normalization: {:?}", xform_ret_ty);

                    // Check whether the impl imposes obligations we have to worry about.
                    let impl_def_id = probe.item.container_id(self.tcx);
                    let impl_bounds = self.tcx.predicates_of(impl_def_id);
                    let impl_bounds = impl_bounds.instantiate(self.tcx, substs);

                    let InferOk { value: impl_bounds, obligations: norm_obligations } =
                        self.fcx.at(&cause, self.param_env).normalize(impl_bounds);

                    // Convert the bounds into obligations.
                    let impl_obligations = traits::predicates_for_generics(
                        |_idx, span| {
                            let misc = traits::ObligationCause::misc(span, self.body_id);
                            let parent_trait_pred = ty::Binder::dummy(ty::TraitPredicate {
                                trait_ref: ty::TraitRef::from_method(self.tcx, impl_def_id, substs),
                                constness: ty::BoundConstness::NotConst,
                                polarity: ty::ImplPolarity::Positive,
                            });
                            misc.derived_cause(parent_trait_pred, |derived| {
                                traits::ImplDerivedObligation(Box::new(
                                    traits::ImplDerivedObligationCause {
                                        derived,
                                        impl_or_alias_def_id: impl_def_id,
                                        impl_def_predicate_index: None,
                                        span,
                                    },
                                ))
                            })
                        },
                        self.param_env,
                        impl_bounds,
                    );

                    let candidate_obligations = impl_obligations
                        .chain(norm_obligations.into_iter())
                        .chain(ref_obligations.iter().cloned())
                        .chain(normalization_obligations.into_iter());

                    // Evaluate those obligations to see if they might possibly hold.
                    for o in candidate_obligations {
                        let o = self.resolve_vars_if_possible(o);
                        if !self.predicate_may_hold(&o) {
                            result = ProbeResult::NoMatch;
                            let parent_o = o.clone();
                            let implied_obligations =
                                traits::elaborate_obligations(self.tcx, vec![o]);
                            for o in implied_obligations {
                                let parent = if o == parent_o {
                                    None
                                } else {
                                    if o.predicate.to_opt_poly_trait_pred().map(|p| p.def_id())
                                        == self.tcx.lang_items().sized_trait()
                                    {
                                        // We don't care to talk about implicit `Sized` bounds.
                                        continue;
                                    }
                                    Some(parent_o.predicate)
                                };
                                if !self.predicate_may_hold(&o) {
                                    possibly_unsatisfied_predicates.push((
                                        o.predicate,
                                        parent,
                                        Some(o.cause),
                                    ));
                                }
                            }
                        }
                    }
                }

                ObjectCandidate | WhereClauseCandidate(..) => {
                    // These have no additional conditions to check.
                }

                TraitCandidate(trait_ref) => {
                    if let Some(method_name) = self.method_name {
                        // Some trait methods are excluded for arrays before 2021.
                        // (`array.into_iter()` wants a slice iterator for compatibility.)
                        if self_ty.is_array() && !method_name.span.rust_2021() {
                            let trait_def = self.tcx.trait_def(trait_ref.def_id);
                            if trait_def.skip_array_during_method_dispatch {
                                return ProbeResult::NoMatch;
                            }
                        }
                    }
                    let predicate =
                        ty::Binder::dummy(trait_ref).without_const().to_predicate(self.tcx);
                    parent_pred = Some(predicate);
                    let obligation =
                        traits::Obligation::new(self.tcx, cause.clone(), self.param_env, predicate);
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
                                            possibly_unsatisfied_predicates.push((
                                                nested_predicate,
                                                p,
                                                Some(obligation.cause.clone()),
                                            ));
                                        }
                                    }
                                }
                                _ => {
                                    // Some nested subobligation of this predicate
                                    // failed.
                                    let predicate = self.resolve_vars_if_possible(predicate);
                                    possibly_unsatisfied_predicates.push((predicate, None, None));
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
                    possibly_unsatisfied_predicates.push((o.predicate, parent_pred, Some(o.cause)));
                }
            }

            if let ProbeResult::Match = result
                && let Some(return_ty) = self.return_type
                && let Some(mut xform_ret_ty) = xform_ret_ty
            {
                // `xform_ret_ty` has only been normalized for `InherentImplCandidate`.
                // We don't normalize the other candidates for perf/backwards-compat reasons...
                // but `self.return_type` is only set on the diagnostic-path, so we
                // should be okay doing it here.
                if !matches!(probe.kind, InherentImplCandidate(..)) {
                    let InferOk {
                        value: normalized_xform_ret_ty,
                        obligations: normalization_obligations,
                    } = self.fcx.at(&cause, self.param_env).normalize(xform_ret_ty);
                    xform_ret_ty = normalized_xform_ret_ty;
                    debug!("xform_ret_ty after normalization: {:?}", xform_ret_ty);
                    // Evaluate those obligations to see if they might possibly hold.
                    for o in normalization_obligations {
                        let o = self.resolve_vars_if_possible(o);
                        if !self.predicate_may_hold(&o) {
                            result = ProbeResult::NoMatch;
                            possibly_unsatisfied_predicates.push((
                                o.predicate,
                                None,
                                Some(o.cause),
                            ));
                        }
                    }
                }

                debug!(
                    "comparing return_ty {:?} with xform ret ty {:?}",
                    return_ty, xform_ret_ty
                );
                if let ProbeResult::Match = result
                    && self
                    .at(&ObligationCause::dummy(), self.param_env)
                    .sup(return_ty, xform_ret_ty)
                    .is_err()
                {
                    result = ProbeResult::BadReturnType;
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
    /// Example (`tests/ui/method-two-trait-defer-resolution-1.rs`):
    ///
    /// ```ignore (illustrative)
    /// trait Foo { ... }
    /// impl Foo for Vec<i32> { ... }
    /// impl Foo for Vec<usize> { ... }
    /// ```
    ///
    /// Now imagine the receiver is `Vec<_>`. It doesn't really matter at this time which impl we
    /// use, so it's ok to just commit to "using the method from the trait Foo".
    fn collapse_candidates_to_trait_pick(
        &self,
        self_ty: Ty<'tcx>,
        probes: &[(&Candidate<'tcx>, ProbeResult)],
    ) -> Option<Pick<'tcx>> {
        // Do all probes correspond to the same trait?
        let container = probes[0].0.item.trait_container(self.tcx)?;
        for (p, _) in &probes[1..] {
            let p_container = p.item.trait_container(self.tcx)?;
            if p_container != container {
                return None;
            }
        }

        // FIXME: check the return type here somehow.
        // If so, just use this trait and call it a day.
        Some(Pick {
            item: probes[0].0.item,
            kind: TraitPick,
            import_ids: probes[0].0.import_ids.clone(),
            autoderefs: 0,
            autoref_or_ptr_adjustment: None,
            self_ty,
            unstable_candidates: vec![],
        })
    }

    /// Similarly to `probe_for_return_type`, this method attempts to find the best matching
    /// candidate method where the method name may have been misspelled. Similarly to other
    /// edit distance based suggestions, we provide at most one such suggestion.
    fn probe_for_similar_candidate(&mut self) -> Result<Option<ty::AssocItem>, MethodError<'tcx>> {
        debug!("probing for method names similar to {:?}", self.method_name);

        let steps = self.steps.clone();
        self.probe(|_| {
            let mut pcx = ProbeContext::new(
                self.fcx,
                self.span,
                self.mode,
                self.method_name,
                self.return_type,
                &self.orig_steps_var_values,
                steps,
                self.scope_expr_id,
            );
            pcx.allow_similar_names = true;
            pcx.assemble_inherent_candidates();

            let method_names = pcx.candidate_method_names(|_| true);
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
                        .map(|cand| cand.name)
                        .collect::<Vec<Symbol>>();
                    find_best_match_for_name_with_substrings(
                        &names,
                        self.method_name.unwrap().name,
                        None,
                    )
                }
                .or_else(|| {
                    applicable_close_candidates
                        .iter()
                        .find(|cand| self.matches_by_doc_alias(cand.def_id))
                        .map(|cand| cand.name)
                })
                .unwrap();
                Ok(applicable_close_candidates.into_iter().find(|method| method.name == best_name))
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

    fn record_static_candidate(&self, source: CandidateSource) {
        self.static_candidates.borrow_mut().push(source);
    }

    #[instrument(level = "debug", skip(self))]
    fn xform_self_ty(
        &self,
        item: ty::AssocItem,
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

    #[instrument(level = "debug", skip(self))]
    fn xform_method_sig(&self, method: DefId, substs: SubstsRef<'tcx>) -> ty::FnSig<'tcx> {
        let fn_sig = self.tcx.fn_sig(method);
        debug!(?fn_sig);

        assert!(!substs.has_escaping_bound_vars());

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The substitutions we
        // are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        let generics = self.tcx.generics_of(method);
        assert_eq!(substs.len(), generics.parent_count as usize);

        let xform_fn_sig = if generics.params.is_empty() {
            fn_sig.subst(self.tcx, substs)
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
                        GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                            self.var_for_def(self.span, param)
                        }
                    }
                }
            });
            fn_sig.subst(self.tcx, substs)
        };

        self.erase_late_bound_regions(xform_fn_sig)
    }

    /// Gets the type of an impl and generate substitutions with inference vars.
    fn impl_ty_and_substs(
        &self,
        impl_def_id: DefId,
    ) -> (ty::EarlyBinder<Ty<'tcx>>, SubstsRef<'tcx>) {
        (self.tcx.type_of(impl_def_id), self.fresh_item_substs(impl_def_id))
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
    fn erase_late_bound_regions<T>(&self, value: ty::Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.tcx.erase_late_bound_regions(value)
    }

    /// Determine if the given associated item type is relevant in the current context.
    fn is_relevant_kind_for_mode(&self, kind: ty::AssocKind) -> bool {
        match (self.mode, kind) {
            (Mode::MethodCall, ty::AssocKind::Fn) => true,
            (Mode::Path, ty::AssocKind::Const | ty::AssocKind::Fn) => true,
            _ => false,
        }
    }

    /// Determine if the associated item withe the given DefId matches
    /// the desired name via a doc alias.
    fn matches_by_doc_alias(&self, def_id: DefId) -> bool {
        let Some(name) = self.method_name else { return false; };
        let Some(local_def_id) = def_id.as_local() else { return false; };
        let hir_id = self.fcx.tcx.hir().local_def_id_to_hir_id(local_def_id);
        let attrs = self.fcx.tcx.hir().attrs(hir_id);
        for attr in attrs {
            let sym::doc = attr.name_or_empty() else { continue; };
            let Some(values) = attr.meta_item_list() else { continue; };
            for v in values {
                if v.name_or_empty() != sym::alias {
                    continue;
                }
                if let Some(nested) = v.meta_item_list() {
                    // #[doc(alias("foo", "bar"))]
                    for n in nested {
                        if let Some(lit) = n.lit() && name.as_str() == lit.symbol.as_str() {
                            return true;
                        }
                    }
                } else if let Some(meta) = v.meta_item()
                    && let Some(lit) = meta.name_value_literal()
                    && name.as_str() == lit.symbol.as_str() {
                        // #[doc(alias = "foo")]
                        return true;
                }
            }
        }
        false
    }

    /// Finds the method with the appropriate name (or return type, as the case may be). If
    /// `allow_similar_names` is set, find methods with close-matching names.
    // The length of the returned iterator is nearly always 0 or 1 and this
    // method is fairly hot.
    fn impl_or_trait_item(&self, def_id: DefId) -> SmallVec<[ty::AssocItem; 1]> {
        if let Some(name) = self.method_name {
            if self.allow_similar_names {
                let max_dist = max(name.as_str().len(), 3) / 3;
                self.tcx
                    .associated_items(def_id)
                    .in_definition_order()
                    .filter(|x| {
                        if !self.is_relevant_kind_for_mode(x.kind) {
                            return false;
                        }
                        if self.matches_by_doc_alias(x.def_id) {
                            return true;
                        }
                        match edit_distance_with_substrings(
                            name.as_str(),
                            x.name.as_str(),
                            max_dist,
                        ) {
                            Some(d) => d > 0,
                            None => false,
                        }
                    })
                    .copied()
                    .collect()
            } else {
                self.fcx
                    .associated_value(def_id, name)
                    .filter(|x| self.is_relevant_kind_for_mode(x.kind))
                    .map_or_else(SmallVec::new, |x| SmallVec::from_buf([x]))
            }
        } else {
            self.tcx
                .associated_items(def_id)
                .in_definition_order()
                .filter(|x| self.is_relevant_kind_for_mode(x.kind))
                .copied()
                .collect()
        }
    }
}

impl<'tcx> Candidate<'tcx> {
    fn to_unadjusted_pick(
        &self,
        self_ty: Ty<'tcx>,
        unstable_candidates: Vec<(Candidate<'tcx>, Symbol)>,
    ) -> Pick<'tcx> {
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
            autoref_or_ptr_adjustment: None,
            self_ty,
            unstable_candidates,
        }
    }
}
