//! Candidate assembly and selection in method resolution - where we enumerate all candidates
//! and choose the best one (or, in some IDE scenarios, just enumerate them all).

use std::{cell::RefCell, convert::Infallible, ops::ControlFlow};

use hir_def::{
    AssocItemId, FunctionId, GenericParamId, ImplId, ItemContainerId, TraitId,
    signatures::TraitFlags,
};
use hir_expand::name::Name;
use rustc_ast_ir::Mutability;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    InferTy, TypeVisitableExt, Upcast, Variance,
    elaborate::{self, supertrait_def_ids},
    fast_reject::{DeepRejectCtxt, TreatParams, simplify_type},
    inherent::{AdtDef as _, BoundExistentialPredicates as _, IntoKind, Ty as _},
};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use self::CandidateKind::*;
pub(super) use self::PickKind::*;
use crate::{
    autoderef::Autoderef,
    db::HirDatabase,
    lower::GenericPredicates,
    method_resolution::{
        CandidateId, CandidateSource, InherentImpls, MethodError, MethodResolutionContext,
        simplified_type_module, with_incoherent_inherent_impls,
    },
    next_solver::{
        Binder, Canonical, ClauseKind, DbInterner, FnSig, GenericArg, GenericArgs, Goal, ParamEnv,
        PolyTraitRef, Predicate, Region, SimplifiedType, TraitRef, Ty, TyKind,
        infer::{
            BoundRegionConversionTime, InferCtxt, InferOk,
            canonical::{QueryResponse, canonicalizer::OriginalQueryValues},
            select::{ImplSource, Selection, SelectionResult},
            traits::{Obligation, ObligationCause, PredicateObligation},
        },
        obligation_ctxt::ObligationCtxt,
        util::clauses_as_obligations,
    },
};

struct ProbeContext<'a, 'db, Choice> {
    ctx: &'a MethodResolutionContext<'a, 'db>,
    mode: Mode,

    /// This is the OriginalQueryValues for the steps queries
    /// that are answered in steps.
    orig_steps_var_values: &'a OriginalQueryValues<'db>,
    steps: &'a [CandidateStep<'db>],

    inherent_candidates: Vec<Candidate<'db>>,
    extension_candidates: Vec<Candidate<'db>>,
    impl_dups: FxHashSet<ImplId>,

    /// List of potential private candidates. Will be trimmed to ones that
    /// actually apply and then the result inserted into `private_candidate`
    private_candidates: Vec<Candidate<'db>>,

    /// Collects near misses when the candidate functions are missing a `self` keyword and is only
    /// used for error reporting
    static_candidates: Vec<CandidateSource>,

    choice: Choice,
}

#[derive(Debug)]
pub struct CandidateWithPrivate<'db> {
    pub candidate: Candidate<'db>,
    pub is_visible: bool,
}

#[derive(Debug, Clone)]
pub struct Candidate<'db> {
    pub item: CandidateId,
    pub kind: CandidateKind<'db>,
}

#[derive(Debug, Clone)]
pub enum CandidateKind<'db> {
    InherentImplCandidate { impl_def_id: ImplId, receiver_steps: usize },
    ObjectCandidate(PolyTraitRef<'db>),
    TraitCandidate(PolyTraitRef<'db>),
    WhereClauseCandidate(PolyTraitRef<'db>),
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum ProbeResult {
    NoMatch,
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
    /// Receiver has type `T`, add `&` or `&mut` (if `T` is `mut`), and maybe also "unsize" it.
    /// Unsizing is used to convert a `[T; N]` to `[T]`, which only makes sense when autorefing.
    Autoref {
        mutbl: Mutability,

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

/// Criteria to apply when searching for a given Pick. This is used during
/// the search for potentially shadowed methods to ensure we don't search
/// more candidates than strictly necessary.
#[derive(Debug)]
struct PickConstraintsForShadowed {
    autoderefs: usize,
    receiver_steps: Option<usize>,
    def_id: CandidateId,
}

impl PickConstraintsForShadowed {
    fn may_shadow_based_on_autoderefs(&self, autoderefs: usize) -> bool {
        autoderefs == self.autoderefs
    }

    fn candidate_may_shadow(&self, candidate: &Candidate<'_>) -> bool {
        // An item never shadows itself
        candidate.item != self.def_id
            // and we're only concerned about inherent impls doing the shadowing.
            // Shadowing can only occur if the shadowed is further along
            // the Receiver dereferencing chain than the shadowed.
            && match candidate.kind {
                CandidateKind::InherentImplCandidate { receiver_steps, .. } => match self.receiver_steps {
                    Some(shadowed_receiver_steps) => receiver_steps > shadowed_receiver_steps,
                    _ => false
                },
                _ => false
            }
    }
}

#[derive(Debug, Clone)]
pub struct Pick<'db> {
    pub item: CandidateId,
    pub kind: PickKind<'db>,

    /// Indicates that the source expression should be autoderef'd N times
    /// ```ignore (not-rust)
    /// A = expr | *expr | **expr | ...
    /// ```
    pub autoderefs: usize,

    /// Indicates that we want to add an autoref (and maybe also unsize it), or if the receiver is
    /// `*mut T`, convert it to `*const T`.
    pub autoref_or_ptr_adjustment: Option<AutorefOrPtrAdjustment>,
    pub self_ty: Ty<'db>,

    /// Number of jumps along the `Receiver::Target` chain we followed
    /// to identify this method. Used only for deshadowing errors.
    /// Only applies for inherent impls.
    pub receiver_steps: Option<usize>,

    /// Candidates that were shadowed by supertraits.
    pub shadowed_candidates: Vec<CandidateId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PickKind<'db> {
    InherentImplPick(ImplId),
    ObjectPick(TraitId),
    TraitPick(TraitId),
    WhereClausePick(
        // Trait
        PolyTraitRef<'db>,
    ),
}

pub(crate) type PickResult<'db> = Result<Pick<'db>, MethodError<'db>>;

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

#[derive(Debug, Clone)]
pub struct CandidateStep<'db> {
    pub self_ty: Canonical<'db, QueryResponse<'db, Ty<'db>>>,
    pub self_ty_is_opaque: bool,
    pub autoderefs: usize,
    /// `true` if the type results from a dereference of a raw pointer.
    /// when assembling candidates, we include these steps, but not when
    /// picking methods. This so that if we have `foo: *const Foo` and `Foo` has methods
    /// `fn by_raw_ptr(self: *const Self)` and `fn by_ref(&self)`, then
    /// `foo.by_raw_ptr()` will work and `foo.by_ref()` won't.
    pub from_unsafe_deref: bool,
    pub unsize: bool,
    /// We will generate CandidateSteps which are reachable via a chain
    /// of following `Receiver`. The first 'n' of those will be reachable
    /// by following a chain of 'Deref' instead (since there's a blanket
    /// implementation of Receiver for Deref).
    /// We use the entire set of steps when identifying method candidates
    /// (e.g. identifying relevant `impl` blocks) but only those that are
    /// reachable via Deref when examining what the receiver type can
    /// be converted into by autodereffing.
    pub reachable_via_deref: bool,
}

#[derive(Clone, Debug)]
struct MethodAutoderefStepsResult<'db> {
    /// The valid autoderef steps that could be found by following a chain
    /// of `Receiver<Target=T>` or `Deref<Target=T>` trait implementations.
    pub steps: SmallVec<[CandidateStep<'db>; 3]>,
    /// If Some(T), a type autoderef reported an error on.
    pub opt_bad_ty: Option<MethodAutoderefBadTy<'db>>,
    /// If `true`, `steps` has been truncated due to reaching the
    /// recursion limit.
    pub reached_recursion_limit: bool,
}

#[derive(Debug, Clone)]
struct MethodAutoderefBadTy<'db> {
    pub reached_raw_pointer: bool,
    pub ty: Canonical<'db, QueryResponse<'db, Ty<'db>>>,
}

impl<'a, 'db> MethodResolutionContext<'a, 'db> {
    #[instrument(level = "debug", skip(self))]
    pub fn probe_for_name(&self, mode: Mode, item_name: Name, self_ty: Ty<'db>) -> PickResult<'db> {
        self.probe_op(mode, self_ty, ProbeForNameChoice { private_candidate: None, item_name })
    }

    #[instrument(level = "debug", skip(self))]
    pub fn probe_all(
        &self,
        mode: Mode,
        self_ty: Ty<'db>,
    ) -> impl Iterator<Item = CandidateWithPrivate<'db>> {
        self.probe_op(mode, self_ty, ProbeAllChoice::new()).candidates.into_inner().into_values()
    }

    fn probe_op<Choice: ProbeChoice<'db>>(
        &self,
        mode: Mode,
        self_ty: Ty<'db>,
        choice: Choice,
    ) -> Choice::FinalChoice {
        let mut orig_values = OriginalQueryValues::default();
        let query_input = self.infcx.canonicalize_query(self_ty, &mut orig_values);
        let steps = match mode {
            Mode::MethodCall => self.method_autoderef_steps(&query_input),
            Mode::Path => self.infcx.probe(|_| {
                // Mode::Path - the deref steps is "trivial". This turns
                // our CanonicalQuery into a "trivial" QueryResponse. This
                // is a bit inefficient, but I don't think that writing
                // special handling for this "trivial case" is a good idea.

                let infcx = self.infcx;
                let (self_ty, var_values) = infcx.instantiate_canonical(&query_input);
                debug!(?self_ty, ?query_input, "probe_op: Mode::Path");
                let prev_opaque_entries =
                    self.infcx.inner.borrow_mut().opaque_types().num_entries();
                MethodAutoderefStepsResult {
                    steps: smallvec![CandidateStep {
                        self_ty: self.infcx.make_query_response_ignoring_pending_obligations(
                            var_values,
                            self_ty,
                            prev_opaque_entries
                        ),
                        self_ty_is_opaque: false,
                        autoderefs: 0,
                        from_unsafe_deref: false,
                        unsize: false,
                        reachable_via_deref: true,
                    }],
                    opt_bad_ty: None,
                    reached_recursion_limit: false,
                }
            }),
        };

        if steps.reached_recursion_limit {
            // FIXME: Report an error.
        }

        // If we encountered an `_` type or an error type during autoderef, this is
        // ambiguous.
        if let Some(bad_ty) = &steps.opt_bad_ty {
            if bad_ty.reached_raw_pointer
                && !self.unstable_features.arbitrary_self_types_pointers
                && self.edition.at_least_2018()
            {
                // this case used to be allowed by the compiler,
                // so we do a future-compat lint here for the 2015 edition
                // (see https://github.com/rust-lang/rust/issues/46906)
                // FIXME: Emit the lint.
                // self.tcx.node_span_lint(
                //     lint::builtin::TYVAR_BEHIND_RAW_POINTER,
                //     scope_expr_id,
                //     span,
                //     |lint| {
                //         lint.primary_message("type annotations needed");
                //     },
                // );
            } else {
                // Ended up encountering a type variable when doing autoderef,
                // but it may not be a type variable after processing obligations
                // in our local `FnCtxt`, so don't call `structurally_resolve_type`.
                let ty = &bad_ty.ty;
                let ty = self
                    .infcx
                    .instantiate_query_response_and_region_obligations(
                        &ObligationCause::new(),
                        self.param_env,
                        &orig_values,
                        ty,
                    )
                    .unwrap_or_else(|_| panic!("instantiating {:?} failed?", ty));
                let ty = self.infcx.resolve_vars_if_possible(ty.value);
                match ty.kind() {
                    TyKind::Infer(InferTy::TyVar(_)) => {
                        // FIXME: Report "type annotations needed" error.
                    }
                    TyKind::Error(_) => {}
                    _ => panic!("unexpected bad final type in method autoderef"),
                };
                return Choice::final_choice_from_err(MethodError::ErrorReported);
            }
        }

        debug!("ProbeContext: steps for self_ty={:?} are {:?}", self_ty, steps);

        // this creates one big transaction so that all type variables etc
        // that we create during the probe process are removed later
        self.infcx.probe(|_| {
            let mut probe_cx = ProbeContext::new(self, mode, &orig_values, &steps.steps, choice);

            probe_cx.assemble_inherent_candidates();
            probe_cx.assemble_extension_candidates_for_traits_in_scope();
            Choice::choose(probe_cx)
        })
    }

    fn method_autoderef_steps(
        &self,
        self_ty: &Canonical<'db, Ty<'db>>,
    ) -> MethodAutoderefStepsResult<'db> {
        self.infcx.probe(|_| {
            debug!("method_autoderef_steps({:?})", self_ty);

            // We accept not-yet-defined opaque types in the autoderef
            // chain to support recursive calls. We do error if the final
            // infer var is not an opaque.
            let infcx = self.infcx;
            let (self_ty, inference_vars) = infcx.instantiate_canonical(self_ty);
            let prev_opaque_entries = infcx.inner.borrow_mut().opaque_types().num_entries();

            let self_ty_is_opaque = |ty: Ty<'_>| {
                if let TyKind::Infer(InferTy::TyVar(vid)) = ty.kind() {
                    infcx.has_opaques_with_sub_unified_hidden_type(vid)
                } else {
                    false
                }
            };

            // If arbitrary self types is not enabled, we follow the chain of
            // `Deref<Target=T>`. If arbitrary self types is enabled, we instead
            // follow the chain of `Receiver<Target=T>`, but we also record whether
            // such types are reachable by following the (potentially shorter)
            // chain of `Deref<Target=T>`. We will use the first list when finding
            // potentially relevant function implementations (e.g. relevant impl blocks)
            // but the second list when determining types that the receiver may be
            // converted to, in order to find out which of those methods might actually
            // be callable.
            let mut autoderef_via_deref =
                Autoderef::new(infcx, self.param_env, self_ty).include_raw_pointers();

            let mut reached_raw_pointer = false;
            let arbitrary_self_types_enabled = self.unstable_features.arbitrary_self_types
                || self.unstable_features.arbitrary_self_types_pointers;
            let (mut steps, reached_recursion_limit) = if arbitrary_self_types_enabled {
                let reachable_via_deref =
                    autoderef_via_deref.by_ref().map(|_| true).chain(std::iter::repeat(false));

                let mut autoderef_via_receiver = Autoderef::new(infcx, self.param_env, self_ty)
                    .include_raw_pointers()
                    .use_receiver_trait();
                let steps = autoderef_via_receiver
                    .by_ref()
                    .zip(reachable_via_deref)
                    .map(|((ty, d), reachable_via_deref)| {
                        let step = CandidateStep {
                            self_ty: infcx.make_query_response_ignoring_pending_obligations(
                                inference_vars,
                                ty,
                                prev_opaque_entries,
                            ),
                            self_ty_is_opaque: self_ty_is_opaque(ty),
                            autoderefs: d,
                            from_unsafe_deref: reached_raw_pointer,
                            unsize: false,
                            reachable_via_deref,
                        };
                        if ty.is_raw_ptr() {
                            // all the subsequent steps will be from_unsafe_deref
                            reached_raw_pointer = true;
                        }
                        step
                    })
                    .collect::<SmallVec<[_; _]>>();
                (steps, autoderef_via_receiver.reached_recursion_limit())
            } else {
                let steps = autoderef_via_deref
                    .by_ref()
                    .map(|(ty, d)| {
                        let step = CandidateStep {
                            self_ty: infcx.make_query_response_ignoring_pending_obligations(
                                inference_vars,
                                ty,
                                prev_opaque_entries,
                            ),
                            self_ty_is_opaque: self_ty_is_opaque(ty),
                            autoderefs: d,
                            from_unsafe_deref: reached_raw_pointer,
                            unsize: false,
                            reachable_via_deref: true,
                        };
                        if ty.is_raw_ptr() {
                            // all the subsequent steps will be from_unsafe_deref
                            reached_raw_pointer = true;
                        }
                        step
                    })
                    .collect();
                (steps, autoderef_via_deref.reached_recursion_limit())
            };
            let final_ty = autoderef_via_deref.final_ty();
            let opt_bad_ty = match final_ty.kind() {
                TyKind::Infer(InferTy::TyVar(_)) if !self_ty_is_opaque(final_ty) => {
                    Some(MethodAutoderefBadTy {
                        reached_raw_pointer,
                        ty: infcx.make_query_response_ignoring_pending_obligations(
                            inference_vars,
                            final_ty,
                            prev_opaque_entries,
                        ),
                    })
                }
                TyKind::Error(_) => Some(MethodAutoderefBadTy {
                    reached_raw_pointer,
                    ty: infcx.make_query_response_ignoring_pending_obligations(
                        inference_vars,
                        final_ty,
                        prev_opaque_entries,
                    ),
                }),
                TyKind::Array(elem_ty, _) => {
                    let autoderefs = steps.iter().filter(|s| s.reachable_via_deref).count() - 1;
                    steps.push(CandidateStep {
                        self_ty: infcx.make_query_response_ignoring_pending_obligations(
                            inference_vars,
                            Ty::new_slice(infcx.interner, elem_ty),
                            prev_opaque_entries,
                        ),
                        self_ty_is_opaque: false,
                        autoderefs,
                        // this could be from an unsafe deref if we had
                        // a *mut/const [T; N]
                        from_unsafe_deref: reached_raw_pointer,
                        unsize: true,
                        reachable_via_deref: true, // this is always the final type from
                                                   // autoderef_via_deref
                    });

                    None
                }
                _ => None,
            };

            debug!("method_autoderef_steps: steps={:?} opt_bad_ty={:?}", steps, opt_bad_ty);
            MethodAutoderefStepsResult { steps, opt_bad_ty, reached_recursion_limit }
        })
    }
}

trait ProbeChoice<'db>: Sized {
    type Choice;
    type FinalChoice;

    /// Finds the method with the appropriate name (or return type, as the case may be).
    // The length of the returned iterator is nearly always 0 or 1 and this
    // method is fairly hot.
    fn with_impl_or_trait_item<'a>(
        this: &mut ProbeContext<'a, 'db, Self>,
        items: &[(Name, AssocItemId)],
        callback: impl FnMut(&mut ProbeContext<'a, 'db, Self>, CandidateId),
    );

    fn consider_candidates(
        this: &ProbeContext<'_, 'db, Self>,
        self_ty: Ty<'db>,
        candidates: Vec<&Candidate<'db>>,
    ) -> ControlFlow<Self::Choice>;

    fn consider_private_candidates(
        this: &mut ProbeContext<'_, 'db, Self>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    );

    fn map_choice_pick(
        choice: Self::Choice,
        f: impl FnOnce(Pick<'db>) -> Pick<'db>,
    ) -> Self::Choice;

    fn check_by_value_method_shadowing(
        this: &mut ProbeContext<'_, 'db, Self>,
        by_value_pick: &Self::Choice,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Self::Choice>;

    fn check_autorefed_method_shadowing(
        this: &mut ProbeContext<'_, 'db, Self>,
        autoref_pick: &Self::Choice,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Self::Choice>;

    fn final_choice_from_err(err: MethodError<'db>) -> Self::FinalChoice;

    fn choose(this: ProbeContext<'_, 'db, Self>) -> Self::FinalChoice;
}

#[derive(Debug)]
struct ProbeForNameChoice<'db> {
    item_name: Name,

    /// Some(candidate) if there is a private candidate
    private_candidate: Option<Pick<'db>>,
}

impl<'db> ProbeChoice<'db> for ProbeForNameChoice<'db> {
    type Choice = PickResult<'db>;
    type FinalChoice = PickResult<'db>;

    fn with_impl_or_trait_item<'a>(
        this: &mut ProbeContext<'a, 'db, Self>,
        items: &[(Name, AssocItemId)],
        mut callback: impl FnMut(&mut ProbeContext<'a, 'db, Self>, CandidateId),
    ) {
        let item = items
            .iter()
            .filter_map(|(name, id)| {
                let id = match *id {
                    AssocItemId::FunctionId(id) => id.into(),
                    AssocItemId::ConstId(id) => id.into(),
                    AssocItemId::TypeAliasId(_) => return None,
                };
                Some((name, id))
            })
            .find(|(name, _)| **name == this.choice.item_name)
            .map(|(_, id)| id)
            .filter(|id| this.mode == Mode::Path || matches!(id, CandidateId::FunctionId(_)));
        if let Some(item) = item {
            callback(this, item);
        }
    }

    fn consider_candidates(
        this: &ProbeContext<'_, 'db, Self>,
        self_ty: Ty<'db>,
        mut applicable_candidates: Vec<&Candidate<'db>>,
    ) -> ControlFlow<Self::Choice> {
        if applicable_candidates.len() > 1
            && let Some(pick) =
                this.collapse_candidates_to_trait_pick(self_ty, &applicable_candidates)
        {
            return ControlFlow::Break(Ok(pick));
        }

        if applicable_candidates.len() > 1 {
            // We collapse to a subtrait pick *after* filtering unstable candidates
            // to make sure we don't prefer a unstable subtrait method over a stable
            // supertrait method.
            if this.ctx.unstable_features.supertrait_item_shadowing
                && let Some(pick) =
                    this.collapse_candidates_to_subtrait_pick(self_ty, &applicable_candidates)
            {
                return ControlFlow::Break(Ok(pick));
            }

            let sources =
                applicable_candidates.iter().map(|p| this.candidate_source(p, self_ty)).collect();
            return ControlFlow::Break(Err(MethodError::Ambiguity(sources)));
        }

        match applicable_candidates.pop() {
            Some(probe) => ControlFlow::Break(Ok(probe.to_unadjusted_pick(self_ty))),
            None => ControlFlow::Continue(()),
        }
    }

    fn consider_private_candidates(
        this: &mut ProbeContext<'_, 'db, Self>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) {
        if this.choice.private_candidate.is_none()
            && let ControlFlow::Break(Ok(pick)) = this.consider_candidates(
                self_ty,
                instantiate_self_ty_obligations,
                &this.private_candidates,
                None,
            )
        {
            this.choice.private_candidate = Some(pick);
        }
    }

    fn map_choice_pick(
        choice: Self::Choice,
        f: impl FnOnce(Pick<'db>) -> Pick<'db>,
    ) -> Self::Choice {
        choice.map(f)
    }

    fn check_by_value_method_shadowing(
        this: &mut ProbeContext<'_, 'db, Self>,
        by_value_pick: &Self::Choice,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Self::Choice> {
        if let Ok(by_value_pick) = by_value_pick
            && matches!(by_value_pick.kind, PickKind::InherentImplPick(_))
        {
            for mutbl in [Mutability::Not, Mutability::Mut] {
                if let Err(e) = this.check_for_shadowed_autorefd_method(
                    by_value_pick,
                    step,
                    self_ty,
                    instantiate_self_ty_obligations,
                    mutbl,
                ) {
                    return ControlFlow::Break(Err(e));
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn check_autorefed_method_shadowing(
        this: &mut ProbeContext<'_, 'db, Self>,
        autoref_pick: &Self::Choice,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Self::Choice> {
        if let Ok(autoref_pick) = autoref_pick.as_ref() {
            // Check we're not shadowing others
            if matches!(autoref_pick.kind, PickKind::InherentImplPick(_))
                && let Err(e) = this.check_for_shadowed_autorefd_method(
                    autoref_pick,
                    step,
                    self_ty,
                    instantiate_self_ty_obligations,
                    Mutability::Mut,
                )
            {
                return ControlFlow::Break(Err(e));
            }
        }
        ControlFlow::Continue(())
    }

    fn final_choice_from_err(err: MethodError<'db>) -> Self::FinalChoice {
        Err(err)
    }

    fn choose(this: ProbeContext<'_, 'db, Self>) -> Self::FinalChoice {
        this.pick()
    }
}

#[derive(Debug)]
struct ProbeAllChoice<'db> {
    candidates: RefCell<FxHashMap<CandidateId, CandidateWithPrivate<'db>>>,
    considering_visible_candidates: bool,
}

impl ProbeAllChoice<'_> {
    fn new() -> Self {
        Self { candidates: RefCell::default(), considering_visible_candidates: true }
    }
}

impl<'db> ProbeChoice<'db> for ProbeAllChoice<'db> {
    type Choice = Infallible;
    type FinalChoice = Self;

    fn with_impl_or_trait_item<'a>(
        this: &mut ProbeContext<'a, 'db, Self>,
        items: &[(Name, AssocItemId)],
        mut callback: impl FnMut(&mut ProbeContext<'a, 'db, Self>, CandidateId),
    ) {
        let mode = this.mode;
        items
            .iter()
            .filter_map(|(_, id)| is_relevant_kind_for_mode(mode, *id))
            .for_each(|id| callback(this, id));
    }

    fn consider_candidates(
        this: &ProbeContext<'_, 'db, Self>,
        _self_ty: Ty<'db>,
        candidates: Vec<&Candidate<'db>>,
    ) -> ControlFlow<Self::Choice> {
        let is_visible = this.choice.considering_visible_candidates;
        let mut all_candidates = this.choice.candidates.borrow_mut();
        for candidate in candidates {
            // We should not override existing entries, because inherent methods of trait objects (from the principal)
            // are also visited as trait methods, and we want to consider them inherent.
            all_candidates
                .entry(candidate.item)
                .or_insert(CandidateWithPrivate { candidate: candidate.clone(), is_visible });
        }
        ControlFlow::Continue(())
    }

    fn consider_private_candidates(
        this: &mut ProbeContext<'_, 'db, Self>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) {
        this.choice.considering_visible_candidates = false;
        let ControlFlow::Continue(()) = this.consider_candidates(
            self_ty,
            instantiate_self_ty_obligations,
            &this.private_candidates,
            None,
        );
        this.choice.considering_visible_candidates = true;
    }

    fn map_choice_pick(
        choice: Self::Choice,
        _f: impl FnOnce(Pick<'db>) -> Pick<'db>,
    ) -> Self::Choice {
        choice
    }

    fn check_by_value_method_shadowing(
        _this: &mut ProbeContext<'_, 'db, Self>,
        _by_value_pick: &Self::Choice,
        _step: &CandidateStep<'db>,
        _self_ty: Ty<'db>,
        _instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Self::Choice> {
        ControlFlow::Continue(())
    }

    fn check_autorefed_method_shadowing(
        _this: &mut ProbeContext<'_, 'db, Self>,
        _autoref_pick: &Self::Choice,
        _step: &CandidateStep<'db>,
        _self_ty: Ty<'db>,
        _instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Self::Choice> {
        ControlFlow::Continue(())
    }

    fn final_choice_from_err(_err: MethodError<'db>) -> Self::FinalChoice {
        Self::new()
    }

    fn choose(mut this: ProbeContext<'_, 'db, Self>) -> Self::FinalChoice {
        let ControlFlow::Continue(()) = this.pick_all_method();
        this.choice
    }
}

impl<'a, 'db, Choice: ProbeChoice<'db>> ProbeContext<'a, 'db, Choice> {
    fn new(
        ctx: &'a MethodResolutionContext<'a, 'db>,
        mode: Mode,
        orig_steps_var_values: &'a OriginalQueryValues<'db>,
        steps: &'a [CandidateStep<'db>],
        choice: Choice,
    ) -> ProbeContext<'a, 'db, Choice> {
        ProbeContext {
            ctx,
            mode,
            inherent_candidates: Vec::new(),
            extension_candidates: Vec::new(),
            impl_dups: FxHashSet::default(),
            orig_steps_var_values,
            steps,
            private_candidates: Vec::new(),
            static_candidates: Vec::new(),
            choice,
        }
    }

    #[inline]
    fn db(&self) -> &'db dyn HirDatabase {
        self.ctx.infcx.interner.db
    }

    #[inline]
    fn interner(&self) -> DbInterner<'db> {
        self.ctx.infcx.interner
    }

    #[inline]
    fn infcx(&self) -> &'a InferCtxt<'db> {
        self.ctx.infcx
    }

    #[inline]
    fn param_env(&self) -> ParamEnv<'db> {
        self.ctx.param_env
    }

    /// When we're looking up a method by path (UFCS), we relate the receiver
    /// types invariantly. When we are looking up a method by the `.` operator,
    /// we relate them covariantly.
    fn variance(&self) -> Variance {
        match self.mode {
            Mode::MethodCall => Variance::Covariant,
            Mode::Path => Variance::Invariant,
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY

    fn push_candidate(&mut self, candidate: Candidate<'db>, is_inherent: bool) {
        let is_accessible = if is_inherent {
            let candidate_id = match candidate.item {
                CandidateId::FunctionId(id) => id.into(),
                CandidateId::ConstId(id) => id.into(),
            };
            let visibility = self.db().assoc_visibility(candidate_id);
            self.ctx.resolver.is_visible(self.db(), visibility)
        } else {
            true
        };
        if is_accessible {
            if is_inherent {
                self.inherent_candidates.push(candidate);
            } else {
                self.extension_candidates.push(candidate);
            }
        } else {
            self.private_candidates.push(candidate);
        }
    }

    fn assemble_inherent_candidates(&mut self) {
        for step in self.steps.iter() {
            self.assemble_probe(&step.self_ty, step.autoderefs);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn assemble_probe(
        &mut self,
        self_ty: &Canonical<'db, QueryResponse<'db, Ty<'db>>>,
        receiver_steps: usize,
    ) {
        let raw_self_ty = self_ty.value.value;
        match raw_self_ty.kind() {
            TyKind::Dynamic(data, ..) => {
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
                    // Using `instantiate_canonical` on our
                    // `Canonical<QueryResponse<Ty<'db>>>` and then *throwing away* the
                    // `CanonicalVarValues` will exactly give us such a generalization - it
                    // will still match the original object type, but it won't pollute our
                    // type variables in any form, so just do that!
                    let (QueryResponse { value: generalized_self_ty, .. }, _ignored_var_values) =
                        self.infcx().instantiate_canonical(self_ty);

                    self.assemble_inherent_candidates_from_object(generalized_self_ty);
                    self.assemble_inherent_impl_candidates_for_type(
                        &SimplifiedType::Trait(p.def_id().0.into()),
                        receiver_steps,
                    );
                    self.assemble_inherent_candidates_for_incoherent_ty(
                        raw_self_ty,
                        receiver_steps,
                    );
                }
            }
            TyKind::Adt(def, _) => {
                let def_id = def.def_id().0;
                self.assemble_inherent_impl_candidates_for_type(
                    &SimplifiedType::Adt(def_id.into()),
                    receiver_steps,
                );
                self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty, receiver_steps);
            }
            TyKind::Foreign(did) => {
                self.assemble_inherent_impl_candidates_for_type(
                    &SimplifiedType::Foreign(did.0.into()),
                    receiver_steps,
                );
                self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty, receiver_steps);
            }
            TyKind::Param(_) => {
                self.assemble_inherent_candidates_from_param(raw_self_ty);
            }
            TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Str
            | TyKind::Array(..)
            | TyKind::Slice(_)
            | TyKind::RawPtr(_, _)
            | TyKind::Ref(..)
            | TyKind::Never
            | TyKind::Tuple(..) => {
                self.assemble_inherent_candidates_for_incoherent_ty(raw_self_ty, receiver_steps)
            }
            _ => {}
        }
    }

    fn assemble_inherent_candidates_for_incoherent_ty(
        &mut self,
        self_ty: Ty<'db>,
        receiver_steps: usize,
    ) {
        let Some(simp) = simplify_type(self.interner(), self_ty, TreatParams::InstantiateWithInfer)
        else {
            panic!("unexpected incoherent type: {:?}", self_ty)
        };
        with_incoherent_inherent_impls(self.db(), self.ctx.resolver.krate(), &simp, |impls| {
            for &impl_def_id in impls {
                self.assemble_inherent_impl_probe(impl_def_id, receiver_steps);
            }
        });
    }

    fn assemble_inherent_impl_candidates_for_type(
        &mut self,
        self_ty: &SimplifiedType,
        receiver_steps: usize,
    ) {
        let Some(module) = simplified_type_module(self.db(), self_ty) else {
            return;
        };
        InherentImpls::for_each_crate_and_block(
            self.db(),
            module.krate(self.db()),
            module.block(self.db()),
            &mut |impls| {
                for &impl_def_id in impls.for_self_ty(self_ty) {
                    self.assemble_inherent_impl_probe(impl_def_id, receiver_steps);
                }
            },
        );
    }

    #[instrument(level = "debug", skip(self))]
    fn assemble_inherent_impl_probe(&mut self, impl_def_id: ImplId, receiver_steps: usize) {
        if !self.impl_dups.insert(impl_def_id) {
            return; // already visited
        }

        self.with_impl_item(impl_def_id, |this, item| {
            if !this.has_applicable_self(item) {
                // No receiver declared. Not a candidate.
                this.record_static_candidate(CandidateSource::Impl(impl_def_id.into()));
                return;
            }
            this.push_candidate(
                Candidate { item, kind: InherentImplCandidate { impl_def_id, receiver_steps } },
                true,
            );
        });
    }

    #[instrument(level = "debug", skip(self))]
    fn assemble_inherent_candidates_from_object(&mut self, self_ty: Ty<'db>) {
        let principal = match self_ty.kind() {
            TyKind::Dynamic(data, ..) => Some(data),
            _ => None,
        }
        .and_then(|data| data.principal())
        .unwrap_or_else(|| {
            panic!("non-object {:?} in assemble_inherent_candidates_from_object", self_ty)
        });

        // It is illegal to invoke a method on a trait instance that refers to
        // the `Self` type. An [`DynCompatibilityViolation::SupertraitSelf`] error
        // will be reported by `dyn_compatibility.rs` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use a
        // instantiation that replaces `Self` with the object type itself. Hence,
        // a `&self` method will wind up with an argument type like `&dyn Trait`.
        let trait_ref = principal.with_self_ty(self.interner(), self_ty);
        self.assemble_candidates_for_bounds(
            elaborate::supertraits(self.interner(), trait_ref),
            |this, new_trait_ref, item| {
                this.push_candidate(Candidate { item, kind: ObjectCandidate(new_trait_ref) }, true);
            },
        );
    }

    #[instrument(level = "debug", skip(self))]
    fn assemble_inherent_candidates_from_param(&mut self, param_ty: Ty<'db>) {
        debug_assert!(matches!(param_ty.kind(), TyKind::Param(_)));

        let interner = self.interner();

        // We use `DeepRejectCtxt` here which may return false positive on where clauses
        // with alias self types. We need to later on reject these as inherent candidates
        // in `consider_probe`.
        let bounds = self.param_env().clauses.iter().filter_map(|predicate| {
            let bound_predicate = predicate.kind();
            match bound_predicate.skip_binder() {
                ClauseKind::Trait(trait_predicate) => DeepRejectCtxt::relate_rigid_rigid(interner)
                    .types_may_unify(param_ty, trait_predicate.trait_ref.self_ty())
                    .then(|| bound_predicate.rebind(trait_predicate.trait_ref)),
                ClauseKind::RegionOutlives(_)
                | ClauseKind::TypeOutlives(_)
                | ClauseKind::Projection(_)
                | ClauseKind::ConstArgHasType(_, _)
                | ClauseKind::WellFormed(_)
                | ClauseKind::ConstEvaluatable(_)
                | ClauseKind::UnstableFeature(_)
                | ClauseKind::HostEffect(..) => None,
            }
        });

        self.assemble_candidates_for_bounds(bounds, |this, poly_trait_ref, item| {
            this.push_candidate(
                Candidate { item, kind: WhereClauseCandidate(poly_trait_ref) },
                true,
            );
        });
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn assemble_candidates_for_bounds<F>(
        &mut self,
        bounds: impl Iterator<Item = PolyTraitRef<'db>>,
        mut mk_cand: F,
    ) where
        F: for<'b> FnMut(&mut ProbeContext<'b, 'db, Choice>, PolyTraitRef<'db>, CandidateId),
    {
        for bound_trait_ref in bounds {
            debug!("elaborate_bounds(bound_trait_ref={:?})", bound_trait_ref);
            self.with_trait_item(bound_trait_ref.def_id().0, |this, item| {
                if !this.has_applicable_self(item) {
                    this.record_static_candidate(CandidateSource::Trait(
                        bound_trait_ref.def_id().0,
                    ));
                } else {
                    mk_cand(this, bound_trait_ref, item);
                }
            });
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn assemble_extension_candidates_for_traits_in_scope(&mut self) {
        for &trait_did in self.ctx.traits_in_scope {
            self.assemble_extension_candidates_for_trait(trait_did);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn assemble_extension_candidates_for_trait(&mut self, trait_def_id: TraitId) {
        let trait_args = self.infcx().fresh_args_for_item(trait_def_id.into());
        let trait_ref = TraitRef::new_from_args(self.interner(), trait_def_id.into(), trait_args);

        self.with_trait_item(trait_def_id, |this, item| {
            // Check whether `trait_def_id` defines a method with suitable name.
            if !this.has_applicable_self(item) {
                debug!("method has inapplicable self");
                this.record_static_candidate(CandidateSource::Trait(trait_def_id));
                return;
            }
            this.push_candidate(
                Candidate { item, kind: TraitCandidate(Binder::dummy(trait_ref)) },
                false,
            );
        });
    }
}

///////////////////////////////////////////////////////////////////////////
// THE ACTUAL SEARCH
impl<'a, 'db> ProbeContext<'a, 'db, ProbeForNameChoice<'db>> {
    #[instrument(level = "debug", skip(self))]
    fn pick(mut self) -> PickResult<'db> {
        if let Some(r) = self.pick_core() {
            return r;
        }

        debug!("pick: actual search failed, assemble diagnostics");

        if let Some(candidate) = self.choice.private_candidate {
            return Err(MethodError::PrivateMatch(candidate));
        }

        Err(MethodError::NoMatch)
    }

    fn pick_core(&mut self) -> Option<PickResult<'db>> {
        self.pick_all_method().break_value()
    }

    /// Check for cases where arbitrary self types allows shadowing
    /// of methods that might be a compatibility break. Specifically,
    /// we have something like:
    /// ```ignore (illustrative)
    /// struct A;
    /// impl A {
    ///   fn foo(self: &NonNull<A>) {}
    ///      // note this is by reference
    /// }
    /// ```
    /// then we've come along and added this method to `NonNull`:
    /// ```ignore (illustrative)
    ///   fn foo(self)  // note this is by value
    /// ```
    /// Report an error in this case.
    fn check_for_shadowed_autorefd_method(
        &mut self,
        possible_shadower: &Pick<'db>,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
        mutbl: Mutability,
    ) -> Result<(), MethodError<'db>> {
        // The errors emitted by this function are part of
        // the arbitrary self types work, and should not impact
        // other users.
        if !self.ctx.unstable_features.arbitrary_self_types
            && !self.ctx.unstable_features.arbitrary_self_types_pointers
        {
            return Ok(());
        }

        // Set criteria for how we find methods possibly shadowed by 'possible_shadower'
        let pick_constraints = PickConstraintsForShadowed {
            // It's the same `self` type...
            autoderefs: possible_shadower.autoderefs,
            // ... but the method was found in an impl block determined
            // by searching further along the Receiver chain than the other,
            // showing that it's a smart pointer type causing the problem...
            receiver_steps: possible_shadower.receiver_steps,
            // ... and they don't end up pointing to the same item in the
            // first place (could happen with things like blanket impls for T)
            def_id: possible_shadower.item,
        };
        // A note on the autoderefs above. Within pick_by_value_method, an extra
        // autoderef may be applied in order to reborrow a reference with
        // a different lifetime. That seems as though it would break the
        // logic of these constraints, since the number of autoderefs could
        // no longer be used to identify the fundamental type of the receiver.
        // However, this extra autoderef is applied only to by-value calls
        // where the receiver is already a reference. So this situation would
        // only occur in cases where the shadowing looks like this:
        // ```
        // struct A;
        // impl A {
        //   fn foo(self: &&NonNull<A>) {}
        //      // note this is by DOUBLE reference
        // }
        // ```
        // then we've come along and added this method to `NonNull`:
        // ```
        //   fn foo(&self)  // note this is by single reference
        // ```
        // and the call is:
        // ```
        // let bar = NonNull<Foo>;
        // let bar = &foo;
        // bar.foo();
        // ```
        // In these circumstances, the logic is wrong, and we wouldn't spot
        // the shadowing, because the autoderef-based maths wouldn't line up.
        // This is a niche case and we can live without generating an error
        // in the case of such shadowing.
        let potentially_shadowed_pick = self.pick_autorefd_method(
            step,
            self_ty,
            instantiate_self_ty_obligations,
            mutbl,
            Some(&pick_constraints),
        );
        // Look for actual pairs of shadower/shadowed which are
        // the sort of shadowing case we want to avoid. Specifically...
        if let ControlFlow::Break(Ok(possible_shadowed)) = &potentially_shadowed_pick {
            let sources = [possible_shadower, possible_shadowed]
                .into_iter()
                .map(|p| self.candidate_source_from_pick(p))
                .collect();
            return Err(MethodError::Ambiguity(sources));
        }
        Ok(())
    }
}

impl<'a, 'db, Choice: ProbeChoice<'db>> ProbeContext<'a, 'db, Choice> {
    fn pick_all_method(&mut self) -> ControlFlow<Choice::Choice> {
        self.steps
            .iter()
            // At this point we're considering the types to which the receiver can be converted,
            // so we want to follow the `Deref` chain not the `Receiver` chain. Filter out
            // steps which can only be reached by following the (longer) `Receiver` chain.
            .filter(|step| step.reachable_via_deref)
            .filter(|step| {
                debug!("pick_all_method: step={:?}", step);
                // Skip types with type errors (but not const/lifetime errors, which are
                // often spurious due to incomplete const evaluation) and raw pointer derefs.
                !step.self_ty.value.value.references_only_ty_error() && !step.from_unsafe_deref
            })
            .try_for_each(|step| {
                let InferOk { value: self_ty, obligations: instantiate_self_ty_obligations } = self
                    .infcx()
                    .instantiate_query_response_and_region_obligations(
                        &ObligationCause::new(),
                        self.param_env(),
                        self.orig_steps_var_values,
                        &step.self_ty,
                    )
                    .unwrap_or_else(|_| panic!("{:?} was applicable but now isn't?", step.self_ty));

                let by_value_pick =
                    self.pick_by_value_method(step, self_ty, &instantiate_self_ty_obligations);

                // Check for shadowing of a by-reference method by a by-value method (see comments on check_for_shadowing)
                if let ControlFlow::Break(by_value_pick) = by_value_pick {
                    Choice::check_by_value_method_shadowing(
                        self,
                        &by_value_pick,
                        step,
                        self_ty,
                        &instantiate_self_ty_obligations,
                    )?;
                    return ControlFlow::Break(by_value_pick);
                }

                let autoref_pick = self.pick_autorefd_method(
                    step,
                    self_ty,
                    &instantiate_self_ty_obligations,
                    Mutability::Not,
                    None,
                );
                // Check for shadowing of a by-mut-ref method by a by-reference method (see comments on check_for_shadowing)
                if let ControlFlow::Break(autoref_pick) = autoref_pick {
                    Choice::check_autorefed_method_shadowing(
                        self,
                        &autoref_pick,
                        step,
                        self_ty,
                        &instantiate_self_ty_obligations,
                    )?;
                    return ControlFlow::Break(autoref_pick);
                }

                // Note that no shadowing errors are produced from here on,
                // as we consider const ptr methods.
                // We allow new methods that take *mut T to shadow
                // methods which took *const T, so there is no entry in
                // this list for the results of `pick_const_ptr_method`.
                // The reason is that the standard pointer cast method
                // (on a mutable pointer) always already shadows the
                // cast method (on a const pointer). So, if we added
                // `pick_const_ptr_method` to this method, the anti-
                // shadowing algorithm would always complain about
                // the conflict between *const::cast and *mut::cast.
                // In practice therefore this does constrain us:
                // we cannot add new
                //   self: *mut Self
                // methods to types such as NonNull or anything else
                // which implements Receiver, because this might in future
                // shadow existing methods taking
                //   self: *const NonNull<Self>
                // in the pointee. In practice, methods taking raw pointers
                // are rare, and it seems that it should be easily possible
                // to avoid such compatibility breaks.
                // We also don't check for reborrowed pin methods which
                // may be shadowed; these also seem unlikely to occur.
                self.pick_autorefd_method(
                    step,
                    self_ty,
                    &instantiate_self_ty_obligations,
                    Mutability::Mut,
                    None,
                )?;
                self.pick_const_ptr_method(step, self_ty, &instantiate_self_ty_obligations)
            })
    }

    /// For each type `T` in the step list, this attempts to find a method where
    /// the (transformed) self type is exactly `T`. We do however do one
    /// transformation on the adjustment: if we are passing a region pointer in,
    /// we will potentially *reborrow* it to a shorter lifetime. This allows us
    /// to transparently pass `&mut` pointers, in particular, without consuming
    /// them for their entire lifetime.
    fn pick_by_value_method(
        &mut self,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Choice::Choice> {
        if step.unsize {
            return ControlFlow::Continue(());
        }

        self.pick_method(self_ty, instantiate_self_ty_obligations, None).map_break(|r| {
            Choice::map_choice_pick(r, |mut pick| {
                pick.autoderefs = step.autoderefs;

                match step.self_ty.value.value.kind() {
                    // Insert a `&*` or `&mut *` if this is a reference type:
                    TyKind::Ref(_, _, mutbl) => {
                        pick.autoderefs += 1;
                        pick.autoref_or_ptr_adjustment = Some(AutorefOrPtrAdjustment::Autoref {
                            mutbl,
                            unsize: pick.autoref_or_ptr_adjustment.is_some_and(|a| a.get_unsize()),
                        })
                    }

                    _ => (),
                }

                pick
            })
        })
    }

    fn pick_autorefd_method(
        &mut self,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
        mutbl: Mutability,
        pick_constraints: Option<&PickConstraintsForShadowed>,
    ) -> ControlFlow<Choice::Choice> {
        let interner = self.interner();

        if let Some(pick_constraints) = pick_constraints
            && !pick_constraints.may_shadow_based_on_autoderefs(step.autoderefs)
        {
            return ControlFlow::Continue(());
        }

        // In general, during probing we erase regions.
        let region = Region::new_erased(interner);

        let autoref_ty = Ty::new_ref(interner, region, self_ty, mutbl);
        self.pick_method(autoref_ty, instantiate_self_ty_obligations, pick_constraints).map_break(
            |r| {
                Choice::map_choice_pick(r, |mut pick| {
                    pick.autoderefs = step.autoderefs;
                    pick.autoref_or_ptr_adjustment =
                        Some(AutorefOrPtrAdjustment::Autoref { mutbl, unsize: step.unsize });
                    pick
                })
            },
        )
    }

    /// If `self_ty` is `*mut T` then this picks `*const T` methods. The reason why we have a
    /// special case for this is because going from `*mut T` to `*const T` with autoderefs and
    /// autorefs would require dereferencing the pointer, which is not safe.
    fn pick_const_ptr_method(
        &mut self,
        step: &CandidateStep<'db>,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
    ) -> ControlFlow<Choice::Choice> {
        // Don't convert an unsized reference to ptr
        if step.unsize {
            return ControlFlow::Continue(());
        }

        let TyKind::RawPtr(ty, Mutability::Mut) = self_ty.kind() else {
            return ControlFlow::Continue(());
        };

        let const_ptr_ty = Ty::new_ptr(self.interner(), ty, Mutability::Not);
        self.pick_method(const_ptr_ty, instantiate_self_ty_obligations, None).map_break(|r| {
            Choice::map_choice_pick(r, |mut pick| {
                pick.autoderefs = step.autoderefs;
                pick.autoref_or_ptr_adjustment = Some(AutorefOrPtrAdjustment::ToConstPtr);
                pick
            })
        })
    }

    fn pick_method(
        &mut self,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
        pick_constraints: Option<&PickConstraintsForShadowed>,
    ) -> ControlFlow<Choice::Choice> {
        debug!("pick_method(self_ty={:?})", self_ty);

        for (kind, candidates) in
            [("inherent", &self.inherent_candidates), ("extension", &self.extension_candidates)]
        {
            debug!("searching {} candidates", kind);
            self.consider_candidates(
                self_ty,
                instantiate_self_ty_obligations,
                candidates,
                pick_constraints,
            )?;
        }

        Choice::consider_private_candidates(self, self_ty, instantiate_self_ty_obligations);

        ControlFlow::Continue(())
    }

    fn consider_candidates(
        &self,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
        candidates: &[Candidate<'db>],
        pick_constraints: Option<&PickConstraintsForShadowed>,
    ) -> ControlFlow<Choice::Choice> {
        let applicable_candidates: Vec<_> = candidates
            .iter()
            .filter(|candidate| {
                pick_constraints
                    .map(|pick_constraints| pick_constraints.candidate_may_shadow(candidate))
                    .unwrap_or(true)
            })
            .filter(|probe| {
                self.consider_probe(self_ty, instantiate_self_ty_obligations, probe)
                    != ProbeResult::NoMatch
            })
            .collect();

        debug!("applicable_candidates: {:?}", applicable_candidates);

        Choice::consider_candidates(self, self_ty, applicable_candidates)
    }

    fn select_trait_candidate(
        &self,
        trait_ref: TraitRef<'db>,
    ) -> SelectionResult<'db, Selection<'db>> {
        let obligation =
            Obligation::new(self.interner(), ObligationCause::new(), self.param_env(), trait_ref);
        self.infcx().select(&obligation)
    }

    /// Used for ambiguous method call error reporting. Uses probing that throws away the result internally,
    /// so do not use to make a decision that may lead to a successful compilation.
    fn candidate_source(&self, candidate: &Candidate<'db>, self_ty: Ty<'db>) -> CandidateSource {
        match candidate.kind {
            InherentImplCandidate { impl_def_id, .. } => CandidateSource::Impl(impl_def_id.into()),
            ObjectCandidate(trait_ref) | WhereClauseCandidate(trait_ref) => {
                CandidateSource::Trait(trait_ref.def_id().0)
            }
            TraitCandidate(trait_ref) => self.infcx().probe(|_| {
                let trait_ref = self.infcx().instantiate_binder_with_fresh_vars(
                    BoundRegionConversionTime::FnCall,
                    trait_ref,
                );
                let (xform_self_ty, _) = self.xform_self_ty(
                    candidate.item,
                    trait_ref.self_ty(),
                    trait_ref.args.as_slice(),
                );
                // Guide the trait selection to show impls that have methods whose type matches
                // up with the `self` parameter of the method.
                let _ = self
                    .infcx()
                    .at(&ObligationCause::dummy(), self.param_env())
                    .sup(xform_self_ty, self_ty);
                match self.select_trait_candidate(trait_ref) {
                    Ok(Some(ImplSource::UserDefined(ref impl_data))) => {
                        // If only a single impl matches, make the error message point
                        // to that impl.
                        CandidateSource::Impl(impl_data.impl_def_id)
                    }
                    _ => CandidateSource::Trait(trait_ref.def_id.0),
                }
            }),
        }
    }

    fn candidate_source_from_pick(&self, pick: &Pick<'db>) -> CandidateSource {
        match pick.kind {
            InherentImplPick(impl_) => CandidateSource::Impl(impl_.into()),
            ObjectPick(trait_) | TraitPick(trait_) => CandidateSource::Trait(trait_),
            WhereClausePick(trait_ref) => CandidateSource::Trait(trait_ref.skip_binder().def_id.0),
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn consider_probe(
        &self,
        self_ty: Ty<'db>,
        instantiate_self_ty_obligations: &[PredicateObligation<'db>],
        probe: &Candidate<'db>,
    ) -> ProbeResult {
        self.infcx().probe(|_| {
            let mut result = ProbeResult::Match;
            let cause = &ObligationCause::new();
            let mut ocx = ObligationCtxt::new(self.infcx());

            // Subtle: we're not *really* instantiating the current self type while
            // probing, but instead fully recompute the autoderef steps once we've got
            // a final `Pick`. We can't nicely handle these obligations outside of a probe.
            //
            // We simply handle them for each candidate here for now. That's kinda scuffed
            // and ideally we just put them into the `FnCtxt` right away. We need to consider
            // them to deal with defining uses in `method_autoderef_steps`.
            ocx.register_obligations(instantiate_self_ty_obligations.iter().cloned());
            let errors = ocx.try_evaluate_obligations();
            if !errors.is_empty() {
                unreachable!("unexpected autoderef error {errors:?}");
            }

            let mut trait_predicate = None;
            let (xform_self_ty, xform_ret_ty);

            match probe.kind {
                InherentImplCandidate { impl_def_id, .. } => {
                    let impl_args = self.infcx().fresh_args_for_item(impl_def_id.into());
                    let impl_ty =
                        self.db().impl_self_ty(impl_def_id).instantiate(self.interner(), impl_args);
                    (xform_self_ty, xform_ret_ty) =
                        self.xform_self_ty(probe.item, impl_ty, impl_args.as_slice());
                    match ocx.relate(
                        cause,
                        self.param_env(),
                        self.variance(),
                        self_ty,
                        xform_self_ty,
                    ) {
                        Ok(()) => {}
                        Err(err) => {
                            debug!("--> cannot relate self-types {:?}", err);
                            return ProbeResult::NoMatch;
                        }
                    }
                    // Check whether the impl imposes obligations we have to worry about.
                    let impl_bounds = GenericPredicates::query_all(self.db(), impl_def_id.into());
                    let impl_bounds = clauses_as_obligations(
                        impl_bounds.iter_instantiated_copied(self.interner(), impl_args.as_slice()),
                        ObligationCause::new(),
                        self.param_env(),
                    );
                    // Convert the bounds into obligations.
                    ocx.register_obligations(impl_bounds);
                }
                TraitCandidate(poly_trait_ref) => {
                    // Some trait methods are excluded for arrays before 2021.
                    // (`array.into_iter()` wants a slice iterator for compatibility.)
                    if self_ty.is_array() && !self.ctx.edition.at_least_2021() {
                        let trait_signature = self.db().trait_signature(poly_trait_ref.def_id().0);
                        if trait_signature
                            .flags
                            .contains(TraitFlags::SKIP_ARRAY_DURING_METHOD_DISPATCH)
                        {
                            return ProbeResult::NoMatch;
                        }
                    }

                    // Some trait methods are excluded for boxed slices before 2024.
                    // (`boxed_slice.into_iter()` wants a slice iterator for compatibility.)
                    if self_ty.boxed_ty().is_some_and(Ty::is_slice)
                        && !self.ctx.edition.at_least_2024()
                    {
                        let trait_signature = self.db().trait_signature(poly_trait_ref.def_id().0);
                        if trait_signature
                            .flags
                            .contains(TraitFlags::SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH)
                        {
                            return ProbeResult::NoMatch;
                        }
                    }

                    let trait_ref = self.infcx().instantiate_binder_with_fresh_vars(
                        BoundRegionConversionTime::FnCall,
                        poly_trait_ref,
                    );
                    (xform_self_ty, xform_ret_ty) = self.xform_self_ty(
                        probe.item,
                        trait_ref.self_ty(),
                        trait_ref.args.as_slice(),
                    );
                    match ocx.relate(
                        cause,
                        self.param_env(),
                        self.variance(),
                        self_ty,
                        xform_self_ty,
                    ) {
                        Ok(()) => {}
                        Err(err) => {
                            debug!("--> cannot relate self-types {:?}", err);
                            return ProbeResult::NoMatch;
                        }
                    }
                    let obligation = Obligation::new(
                        self.interner(),
                        cause.clone(),
                        self.param_env(),
                        Binder::dummy(trait_ref),
                    );

                    // We only need this hack to deal with fatal overflow in the old solver.
                    ocx.register_obligation(obligation);

                    trait_predicate = Some(trait_ref.upcast(self.interner()));
                }
                ObjectCandidate(poly_trait_ref) | WhereClauseCandidate(poly_trait_ref) => {
                    let trait_ref = self.infcx().instantiate_binder_with_fresh_vars(
                        BoundRegionConversionTime::FnCall,
                        poly_trait_ref,
                    );
                    (xform_self_ty, xform_ret_ty) = self.xform_self_ty(
                        probe.item,
                        trait_ref.self_ty(),
                        trait_ref.args.as_slice(),
                    );

                    if matches!(probe.kind, WhereClauseCandidate(_)) {
                        // `WhereClauseCandidate` requires that the self type is a param,
                        // because it has special behavior with candidate preference as an
                        // inherent pick.
                        match ocx.structurally_normalize_ty(
                            cause,
                            self.param_env(),
                            trait_ref.self_ty(),
                        ) {
                            Ok(ty) => {
                                if !matches!(ty.kind(), TyKind::Param(_)) {
                                    debug!("--> not a param ty: {xform_self_ty:?}");
                                    return ProbeResult::NoMatch;
                                }
                            }
                            Err(errors) => {
                                debug!("--> cannot relate self-types {:?}", errors);
                                return ProbeResult::NoMatch;
                            }
                        }
                    }

                    match ocx.relate(
                        cause,
                        self.param_env(),
                        self.variance(),
                        self_ty,
                        xform_self_ty,
                    ) {
                        Ok(()) => {}
                        Err(err) => {
                            debug!("--> cannot relate self-types {:?}", err);
                            return ProbeResult::NoMatch;
                        }
                    }
                }
            }

            // See <https://github.com/rust-lang/trait-system-refactor-initiative/issues/134>.
            //
            // In the new solver, check the well-formedness of the return type.
            // This emulates, in a way, the predicates that fall out of
            // normalizing the return type in the old solver.
            //
            // FIXME(-Znext-solver): We alternatively could check the predicates of
            // the method itself hold, but we intentionally do not do this in the old
            // solver b/c of cycles, and doing it in the new solver would be stronger.
            // This should be fixed in the future, since it likely leads to much better
            // method winnowing.
            if let Some(xform_ret_ty) = xform_ret_ty {
                ocx.register_obligation(Obligation::new(
                    self.interner(),
                    cause.clone(),
                    self.param_env(),
                    ClauseKind::WellFormed(xform_ret_ty.into()),
                ));
            }

            if !ocx.try_evaluate_obligations().is_empty() {
                result = ProbeResult::NoMatch;
            }

            if self.should_reject_candidate_due_to_opaque_treated_as_rigid(trait_predicate) {
                result = ProbeResult::NoMatch;
            }

            // FIXME: Need to leak-check here.
            // if let Err(_) = self.leak_check(outer_universe, Some(snapshot)) {
            //     result = ProbeResult::NoMatch;
            // }

            result
        })
    }

    /// Trait candidates for not-yet-defined opaque types are a somewhat hacky.
    ///
    /// We want to only accept trait methods if they were hold even if the
    /// opaque types were rigid. To handle this, we both check that for trait
    /// candidates the goal were to hold even when treating opaques as rigid,
    /// see `rustc_trait_selection::solve::OpaqueTypesJank`.
    ///
    /// We also check that all opaque types encountered as self types in the
    /// autoderef chain don't get constrained when applying the candidate.
    /// Importantly, this also handles calling methods taking `&self` on
    /// `impl Trait` to reject the "by-self" candidate.
    ///
    /// This needs to happen at the end of `consider_probe` as we need to take
    /// all the constraints from that into account.
    #[instrument(level = "debug", skip(self), ret)]
    fn should_reject_candidate_due_to_opaque_treated_as_rigid(
        &self,
        trait_predicate: Option<Predicate<'db>>,
    ) -> bool {
        // This function is what hacky and doesn't perfectly do what we want it to.
        // It's not soundness critical and we should be able to freely improve this
        // in the future.
        //
        // Some concrete edge cases include the fact that `goal_may_hold_opaque_types_jank`
        // also fails if there are any constraints opaques which are never used as a self
        // type. We also allow where-bounds which are currently ambiguous but end up
        // constraining an opaque later on.

        // Check whether the trait candidate would not be applicable if the
        // opaque type were rigid.
        if let Some(predicate) = trait_predicate {
            let goal = Goal { param_env: self.param_env(), predicate };
            if !self.infcx().goal_may_hold_opaque_types_jank(goal) {
                return true;
            }
        }

        // Check whether any opaque types in the autoderef chain have been
        // constrained.
        for step in self.steps {
            if step.self_ty_is_opaque {
                debug!(?step.autoderefs, ?step.self_ty, "self_type_is_opaque");
                let constrained_opaque = self.infcx().probe(|_| {
                    // If we fail to instantiate the self type of this
                    // step, this part of the deref-chain is no longer
                    // reachable. In this case we don't care about opaque
                    // types there.
                    let Ok(ok) = self.infcx().instantiate_query_response_and_region_obligations(
                        &ObligationCause::new(),
                        self.param_env(),
                        self.orig_steps_var_values,
                        &step.self_ty,
                    ) else {
                        debug!("failed to instantiate self_ty");
                        return false;
                    };
                    let mut ocx = ObligationCtxt::new(self.infcx());
                    let self_ty = ocx.register_infer_ok_obligations(ok);
                    if !ocx.try_evaluate_obligations().is_empty() {
                        debug!("failed to prove instantiate self_ty obligations");
                        return false;
                    }

                    !self.infcx().resolve_vars_if_possible(self_ty).is_ty_var()
                });
                if constrained_opaque {
                    debug!("opaque type has been constrained");
                    return true;
                }
            }
        }

        false
    }

    /// Sometimes we get in a situation where we have multiple probes that are all impls of the
    /// same trait, but we don't know which impl to use. In this case, since in all cases the
    /// external interface of the method can be determined from the trait, it's ok not to decide.
    /// We can basically just collapse all of the probes for various impls into one where-clause
    /// probe. This will result in a pending obligation so when more type-info is available we can
    /// make the final decision.
    ///
    /// Example (`tests/ui/methods/method-two-trait-defer-resolution-1.rs`):
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
        self_ty: Ty<'db>,
        probes: &[&Candidate<'db>],
    ) -> Option<Pick<'db>> {
        // Do all probes correspond to the same trait?
        let ItemContainerId::TraitId(container) = probes[0].item.container(self.db()) else {
            return None;
        };
        for p in &probes[1..] {
            let ItemContainerId::TraitId(p_container) = p.item.container(self.db()) else {
                return None;
            };
            if p_container != container {
                return None;
            }
        }

        // FIXME: check the return type here somehow.
        // If so, just use this trait and call it a day.
        Some(Pick {
            item: probes[0].item,
            kind: TraitPick(container),
            autoderefs: 0,
            autoref_or_ptr_adjustment: None,
            self_ty,
            receiver_steps: None,
            shadowed_candidates: vec![],
        })
    }

    /// Much like `collapse_candidates_to_trait_pick`, this method allows us to collapse
    /// multiple conflicting picks if there is one pick whose trait container is a subtrait
    /// of the trait containers of all of the other picks.
    ///
    /// This implements RFC #3624.
    fn collapse_candidates_to_subtrait_pick(
        &self,
        self_ty: Ty<'db>,
        probes: &[&Candidate<'db>],
    ) -> Option<Pick<'db>> {
        let mut child_candidate = probes[0];
        let ItemContainerId::TraitId(mut child_trait) = child_candidate.item.container(self.db())
        else {
            return None;
        };
        let mut supertraits: FxHashSet<_> =
            supertrait_def_ids(self.interner(), child_trait.into()).collect();

        let mut remaining_candidates: Vec<_> = probes[1..].to_vec();
        while !remaining_candidates.is_empty() {
            let mut made_progress = false;
            let mut next_round = vec![];

            for remaining_candidate in remaining_candidates {
                let ItemContainerId::TraitId(remaining_trait) =
                    remaining_candidate.item.container(self.db())
                else {
                    return None;
                };
                if supertraits.contains(&remaining_trait.into()) {
                    made_progress = true;
                    continue;
                }

                // This pick is not a supertrait of the `child_pick`.
                // Check if it's a subtrait of the `child_pick`, instead.
                // If it is, then it must have been a subtrait of every
                // other pick we've eliminated at this point. It will
                // take over at this point.
                let remaining_trait_supertraits: FxHashSet<_> =
                    supertrait_def_ids(self.interner(), remaining_trait.into()).collect();
                if remaining_trait_supertraits.contains(&child_trait.into()) {
                    child_candidate = remaining_candidate;
                    child_trait = remaining_trait;
                    supertraits = remaining_trait_supertraits;
                    made_progress = true;
                    continue;
                }

                // `child_pick` is not a supertrait of this pick.
                // Don't bail here, since we may be comparing two supertraits
                // of a common subtrait. These two supertraits won't be related
                // at all, but we will pick them up next round when we find their
                // child as we continue iterating in this round.
                next_round.push(remaining_candidate);
            }

            if made_progress {
                // If we've made progress, iterate again.
                remaining_candidates = next_round;
            } else {
                // Otherwise, we must have at least two candidates which
                // are not related to each other at all.
                return None;
            }
        }

        Some(Pick {
            item: child_candidate.item,
            kind: TraitPick(child_trait),
            autoderefs: 0,
            autoref_or_ptr_adjustment: None,
            self_ty,
            shadowed_candidates: probes
                .iter()
                .map(|c| c.item)
                .filter(|item| *item != child_candidate.item)
                .collect(),
            receiver_steps: None,
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY
    fn has_applicable_self(&self, item: CandidateId) -> bool {
        // "Fast track" -- check for usage of sugar when in method call
        // mode.
        //
        // In Path mode (i.e., resolving a value like `T::next`), consider any
        // associated value (i.e., methods, constants).
        match item {
            CandidateId::FunctionId(id) if self.mode == Mode::MethodCall => {
                self.db().function_signature(id).has_self_param()
            }
            _ => true,
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

    #[instrument(level = "debug", skip(self))]
    fn xform_self_ty(
        &self,
        item: CandidateId,
        impl_ty: Ty<'db>,
        args: &[GenericArg<'db>],
    ) -> (Ty<'db>, Option<Ty<'db>>) {
        if let CandidateId::FunctionId(item) = item
            && self.mode == Mode::MethodCall
        {
            let sig = self.xform_method_sig(item, args);
            (sig.inputs()[0], Some(sig.output()))
        } else {
            (impl_ty, None)
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn xform_method_sig(&self, method: FunctionId, args: &[GenericArg<'db>]) -> FnSig<'db> {
        let fn_sig = self.db().callable_item_signature(method.into());
        debug!(?fn_sig);

        assert!(!args.has_escaping_bound_vars());

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The generic parameters
        // we are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        let generics = self.db().generic_params(method.into());

        let xform_fn_sig = if generics.is_empty() {
            fn_sig.instantiate(self.interner(), args)
        } else {
            let args = GenericArgs::for_item(
                self.interner(),
                method.into(),
                |param_index, param_id, _| {
                    let i = param_index as usize;
                    if i < args.len() {
                        args[i]
                    } else {
                        match param_id {
                            GenericParamId::LifetimeParamId(_) => {
                                // In general, during probe we erase regions.
                                Region::new_erased(self.interner()).into()
                            }
                            GenericParamId::TypeParamId(_) => self.infcx().next_ty_var().into(),
                            GenericParamId::ConstParamId(_) => self.infcx().next_const_var().into(),
                        }
                    }
                },
            );
            fn_sig.instantiate(self.interner(), args)
        };

        self.interner().instantiate_bound_regions_with_erased(xform_fn_sig)
    }

    fn with_impl_item(&mut self, def_id: ImplId, callback: impl FnMut(&mut Self, CandidateId)) {
        Choice::with_impl_or_trait_item(self, &def_id.impl_items(self.db()).items, callback)
    }

    fn with_trait_item(&mut self, def_id: TraitId, callback: impl FnMut(&mut Self, CandidateId)) {
        Choice::with_impl_or_trait_item(self, &def_id.trait_items(self.db()).items, callback)
    }
}

/// Determine if the given associated item type is relevant in the current context.
fn is_relevant_kind_for_mode(mode: Mode, kind: AssocItemId) -> Option<CandidateId> {
    Some(match (mode, kind) {
        (Mode::MethodCall, AssocItemId::FunctionId(id)) => id.into(),
        (Mode::Path, AssocItemId::ConstId(id)) => id.into(),
        (Mode::Path, AssocItemId::FunctionId(id)) => id.into(),
        _ => return None,
    })
}

impl<'db> Candidate<'db> {
    fn to_unadjusted_pick(&self, self_ty: Ty<'db>) -> Pick<'db> {
        Pick {
            item: self.item,
            kind: match self.kind {
                InherentImplCandidate { impl_def_id, .. } => InherentImplPick(impl_def_id),
                ObjectCandidate(trait_ref) => ObjectPick(trait_ref.skip_binder().def_id.0),
                TraitCandidate(trait_ref) => TraitPick(trait_ref.skip_binder().def_id.0),
                WhereClauseCandidate(trait_ref) => {
                    // Only trait derived from where-clauses should
                    // appear here, so they should not contain any
                    // inference variables or other artifacts. This
                    // means they are safe to put into the
                    // `WhereClausePick`.
                    assert!(
                        !trait_ref.skip_binder().args.has_infer()
                            && !trait_ref.skip_binder().args.has_placeholders()
                    );

                    WhereClausePick(trait_ref)
                }
            },
            autoderefs: 0,
            autoref_or_ptr_adjustment: None,
            self_ty,
            receiver_steps: match self.kind {
                InherentImplCandidate { receiver_steps, .. } => Some(receiver_steps),
                _ => None,
            },
            shadowed_candidates: vec![],
        }
    }
}
