//! Code for projecting associated types out of trait references.

use super::elaborate_predicates;
use super::specialization_graph;
use super::translate_substs;
use super::Obligation;
use super::ObligationCause;
use super::PredicateObligation;
use super::Selection;
use super::SelectionContext;
use super::SelectionError;
use super::{VtableImplData, VtableClosureData, VtableGeneratorData, VtableFnPointerData};
use super::util;

use crate::hir::def_id::DefId;
use crate::infer::{InferCtxt, InferOk, LateBoundRegionConversionTime};
use crate::infer::type_variable::TypeVariableOrigin;
use crate::mir::interpret::{GlobalId};
use rustc_data_structures::snapshot_map::{Snapshot, SnapshotMap};
use syntax::ast::Ident;
use crate::ty::subst::{Subst, InternalSubsts};
use crate::ty::{self, ToPredicate, ToPolyTraitRef, Ty, TyCtxt};
use crate::ty::fold::{TypeFoldable, TypeFolder};
use crate::util::common::FN_OUTPUT_NAME;

/// Depending on the stage of compilation, we want projection to be
/// more or less conservative.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Reveal {
    /// At type-checking time, we refuse to project any associated
    /// type that is marked `default`. Non-`default` ("final") types
    /// are always projected. This is necessary in general for
    /// soundness of specialization. However, we *could* allow
    /// projections in fully-monomorphic cases. We choose not to,
    /// because we prefer for `default type` to force the type
    /// definition to be treated abstractly by any consumers of the
    /// impl. Concretely, that means that the following example will
    /// fail to compile:
    ///
    /// ```
    /// trait Assoc {
    ///     type Output;
    /// }
    ///
    /// impl<T> Assoc for T {
    ///     default type Output = bool;
    /// }
    ///
    /// fn main() {
    ///     let <() as Assoc>::Output = true;
    /// }
    UserFacing,

    /// At codegen time, all monomorphic projections will succeed.
    /// Also, `impl Trait` is normalized to the concrete type,
    /// which has to be already collected by type-checking.
    ///
    /// NOTE: as `impl Trait`'s concrete type should *never*
    /// be observable directly by the user, `Reveal::All`
    /// should not be used by checks which may expose
    /// type equality or type contents to the user.
    /// There are some exceptions, e.g., around OIBITS and
    /// transmute-checking, which expose some details, but
    /// not the whole concrete type of the `impl Trait`.
    All,
}

pub type PolyProjectionObligation<'tcx> =
    Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTyObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionTy<'tcx>>;

/// When attempting to resolve `<T as TraitRef>::Name` ...
#[derive(Debug)]
pub enum ProjectionTyError<'tcx> {
    /// ...we found multiple sources of information and couldn't resolve the ambiguity.
    TooManyCandidates,

    /// ...an error occurred matching `T : TraitRef`
    TraitSelectionError(SelectionError<'tcx>),
}

#[derive(Clone)]
pub struct MismatchedProjectionTypes<'tcx> {
    pub err: ty::error::TypeError<'tcx>
}

#[derive(PartialEq, Eq, Debug)]
enum ProjectionTyCandidate<'tcx> {
    // from a where-clause in the env or object type
    ParamEnv(ty::PolyProjectionPredicate<'tcx>),

    // from the definition of `Trait` when you have something like <<A as Trait>::B as Trait2>::C
    TraitDef(ty::PolyProjectionPredicate<'tcx>),

    // from a "impl" (or a "pseudo-impl" returned by select)
    Select(Selection<'tcx>),
}

enum ProjectionTyCandidateSet<'tcx> {
    None,
    Single(ProjectionTyCandidate<'tcx>),
    Ambiguous,
    Error(SelectionError<'tcx>),
}

impl<'tcx> ProjectionTyCandidateSet<'tcx> {
    fn mark_ambiguous(&mut self) {
        *self = ProjectionTyCandidateSet::Ambiguous;
    }

    fn mark_error(&mut self, err: SelectionError<'tcx>) {
        *self = ProjectionTyCandidateSet::Error(err);
    }

    // Returns true if the push was successful, or false if the candidate
    // was discarded -- this could be because of ambiguity, or because
    // a higher-priority candidate is already there.
    fn push_candidate(&mut self, candidate: ProjectionTyCandidate<'tcx>) -> bool {
        use self::ProjectionTyCandidateSet::*;
        use self::ProjectionTyCandidate::*;

        // This wacky variable is just used to try and
        // make code readable and avoid confusing paths.
        // It is assigned a "value" of `()` only on those
        // paths in which we wish to convert `*self` to
        // ambiguous (and return false, because the candidate
        // was not used). On other paths, it is not assigned,
        // and hence if those paths *could* reach the code that
        // comes after the match, this fn would not compile.
        let convert_to_ambiguous;

        match self {
            None => {
                *self = Single(candidate);
                return true;
            }

            Single(current) => {
                // Duplicates can happen inside ParamEnv. In the case, we
                // perform a lazy deduplication.
                if current == &candidate {
                    return false;
                }

                // Prefer where-clauses. As in select, if there are multiple
                // candidates, we prefer where-clause candidates over impls.  This
                // may seem a bit surprising, since impls are the source of
                // "truth" in some sense, but in fact some of the impls that SEEM
                // applicable are not, because of nested obligations. Where
                // clauses are the safer choice. See the comment on
                // `select::SelectionCandidate` and #21974 for more details.
                match (current, candidate) {
                    (ParamEnv(..), ParamEnv(..)) => convert_to_ambiguous = (),
                    (ParamEnv(..), _) => return false,
                    (_, ParamEnv(..)) => unreachable!(),
                    (_, _) => convert_to_ambiguous = (),
                }
            }

            Ambiguous | Error(..) => {
                return false;
            }
        }

        // We only ever get here when we moved from a single candidate
        // to ambiguous.
        let () = convert_to_ambiguous;
        *self = Ambiguous;
        false
    }
}

/// Evaluates constraints of the form:
///
///     for<...> <T as Trait>::U == V
///
/// If successful, this may result in additional obligations. Also returns
/// the projection cache key used to track these additional obligations.
pub fn poly_project_and_unify_type<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &PolyProjectionObligation<'tcx>)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>,
              MismatchedProjectionTypes<'tcx>>
{
    debug!("poly_project_and_unify_type(obligation={:?})",
           obligation);

    let infcx = selcx.infcx();
    infcx.commit_if_ok(|snapshot| {
        let (placeholder_predicate, placeholder_map) =
            infcx.replace_bound_vars_with_placeholders(&obligation.predicate);

        let placeholder_obligation = obligation.with(placeholder_predicate);
        let result = project_and_unify_type(selcx, &placeholder_obligation)?;
        infcx.leak_check(false, &placeholder_map, snapshot)
            .map_err(|err| MismatchedProjectionTypes { err })?;
        Ok(result)
    })
}

/// Evaluates constraints of the form:
///
///     <T as Trait>::U == V
///
/// If successful, this may result in additional obligations.
fn project_and_unify_type<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionObligation<'tcx>)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>,
              MismatchedProjectionTypes<'tcx>>
{
    debug!("project_and_unify_type(obligation={:?})",
           obligation);

    let mut obligations = vec![];
    let normalized_ty =
        match opt_normalize_projection_type(selcx,
                                            obligation.param_env,
                                            obligation.predicate.projection_ty,
                                            obligation.cause.clone(),
                                            obligation.recursion_depth,
                                            &mut obligations) {
            Some(n) => n,
            None => return Ok(None),
        };

    debug!("project_and_unify_type: normalized_ty={:?} obligations={:?}",
           normalized_ty,
           obligations);

    let infcx = selcx.infcx();
    match infcx.at(&obligation.cause, obligation.param_env)
               .eq(normalized_ty, obligation.predicate.ty) {
        Ok(InferOk { obligations: inferred_obligations, value: () }) => {
            obligations.extend(inferred_obligations);
            Ok(Some(obligations))
        },
        Err(err) => {
            debug!("project_and_unify_type: equating types encountered error {:?}", err);
            Err(MismatchedProjectionTypes { err })
        }
    }
}

/// Normalizes any associated type projections in `value`, replacing
/// them with a fully resolved type where possible. The return value
/// combines the normalized result and any additional obligations that
/// were incurred as result.
pub fn normalize<'a, 'b, 'gcx, 'tcx, T>(selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
                                        param_env: ty::ParamEnv<'tcx>,
                                        cause: ObligationCause<'tcx>,
                                        value: &T)
                                        -> Normalized<'tcx, T>
    where T : TypeFoldable<'tcx>
{
    normalize_with_depth(selcx, param_env, cause, 0, value)
}

/// As `normalize`, but with a custom depth.
pub fn normalize_with_depth<'a, 'b, 'gcx, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    value: &T)
    -> Normalized<'tcx, T>

    where T : TypeFoldable<'tcx>
{
    debug!("normalize_with_depth(depth={}, value={:?})", depth, value);
    let mut normalizer = AssociatedTypeNormalizer::new(selcx, param_env, cause, depth);
    let result = normalizer.fold(value);
    debug!("normalize_with_depth: depth={} result={:?} with {} obligations",
           depth, result, normalizer.obligations.len());
    debug!("normalize_with_depth: depth={} obligations={:?}",
           depth, normalizer.obligations);
    Normalized {
        value: result,
        obligations: normalizer.obligations,
    }
}

struct AssociatedTypeNormalizer<'a, 'b: 'a, 'gcx: 'b+'tcx, 'tcx: 'b> {
    selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
    depth: usize,
}

impl<'a, 'b, 'gcx, 'tcx> AssociatedTypeNormalizer<'a, 'b, 'gcx, 'tcx> {
    fn new(selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
           param_env: ty::ParamEnv<'tcx>,
           cause: ObligationCause<'tcx>,
           depth: usize)
           -> AssociatedTypeNormalizer<'a, 'b, 'gcx, 'tcx>
    {
        AssociatedTypeNormalizer {
            selcx,
            param_env,
            cause,
            obligations: vec![],
            depth,
        }
    }

    fn fold<T:TypeFoldable<'tcx>>(&mut self, value: &T) -> T {
        let value = self.selcx.infcx().resolve_type_vars_if_possible(value);

        if !value.has_projections() {
            value
        } else {
            value.fold_with(self).unwrap_or_else(|e: !| e)
        }
    }
}

impl<'a, 'b, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for AssociatedTypeNormalizer<'a, 'b, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'gcx, 'tcx> {
        self.selcx.tcx()
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        // We don't want to normalize associated types that occur inside of region
        // binders, because they may contain bound regions, and we can't cope with that.
        //
        // Example:
        //
        //     for<'a> fn(<T as Foo<&'a>>::A)
        //
        // Instead of normalizing `<T as Foo<&'a>>::A` here, we'll
        // normalize it when we instantiate those bound regions (which
        // should occur eventually).

        let ty = ty.super_fold_with(self)?;
        Ok(match ty.sty {
            ty::Opaque(def_id, substs) if !substs.has_escaping_bound_vars() => { // (*)
                // Only normalize `impl Trait` after type-checking, usually in codegen.
                match self.param_env.reveal {
                    Reveal::UserFacing => ty,

                    Reveal::All => {
                        let recursion_limit = *self.tcx().sess.recursion_limit.get();
                        if self.depth >= recursion_limit {
                            let obligation = Obligation::with_depth(
                                self.cause.clone(),
                                recursion_limit,
                                self.param_env,
                                ty,
                            );
                            self.selcx.infcx().report_overflow_error(&obligation, true);
                        }

                        let generic_ty = self.tcx().type_of(def_id);
                        let concrete_ty = generic_ty.subst(self.tcx(), substs);
                        self.depth += 1;
                        let folded_ty = self.fold_ty(concrete_ty)?;
                        self.depth -= 1;
                        folded_ty
                    }
                }
            }

            ty::Projection(ref data) if !data.has_escaping_bound_vars() => { // (*)

                // (*) This is kind of hacky -- we need to be able to
                // handle normalization within binders because
                // otherwise we wind up a need to normalize when doing
                // trait matching (since you can have a trait
                // obligation like `for<'a> T::B : Fn(&'a int)`), but
                // we can't normalize with bound regions in scope. So
                // far now we just ignore binders but only normalize
                // if all bound regions are gone (and then we still
                // have to renormalize whenever we instantiate a
                // binder). It would be better to normalize in a
                // binding-aware fashion.

                let normalized_ty = normalize_projection_type(self.selcx,
                                                              self.param_env,
                                                              data.clone(),
                                                              self.cause.clone(),
                                                              self.depth,
                                                              &mut self.obligations);
                debug!("AssociatedTypeNormalizer: depth={} normalized {:?} to {:?}, \
                        now with {} obligations",
                       self.depth, ty, normalized_ty, self.obligations.len());
                normalized_ty
            }

            _ => ty
        })
    }

    fn fold_const(&mut self, constant: &'tcx ty::LazyConst<'tcx>)
                  -> Result<&'tcx ty::LazyConst<'tcx>, !>
    {
        if let ty::LazyConst::Unevaluated(def_id, substs) = *constant {
            let tcx = self.selcx.tcx().global_tcx();
            if let Some(param_env) = self.tcx().lift_to_global(&self.param_env) {
                if substs.needs_infer() || substs.has_placeholders() {
                    let identity_substs = InternalSubsts::identity_for_item(tcx, def_id);
                    let instance = ty::Instance::resolve(tcx, param_env, def_id, identity_substs);
                    if let Some(instance) = instance {
                        let cid = GlobalId {
                            instance,
                            promoted: None
                        };
                        if let Ok(evaluated) = tcx.const_eval(param_env.and(cid)) {
                            let substs = tcx.lift_to_global(&substs).unwrap();
                            let evaluated = evaluated.subst(tcx, substs);
                            return Ok(tcx.mk_lazy_const(ty::LazyConst::Evaluated(evaluated)));
                        }
                    }
                } else {
                    if let Some(substs) = self.tcx().lift_to_global(&substs) {
                        let instance = ty::Instance::resolve(tcx, param_env, def_id, substs);
                        if let Some(instance) = instance {
                            let cid = GlobalId {
                                instance,
                                promoted: None
                            };
                            if let Ok(evaluated) = tcx.const_eval(param_env.and(cid)) {
                                return Ok(tcx.mk_lazy_const(ty::LazyConst::Evaluated(evaluated)));
                            }
                        }
                    }
                }
            }
        }
        Ok(constant)
    }
}

#[derive(Clone)]
pub struct Normalized<'tcx,T> {
    pub value: T,
    pub obligations: Vec<PredicateObligation<'tcx>>,
}

pub type NormalizedTy<'tcx> = Normalized<'tcx, Ty<'tcx>>;

impl<'tcx,T> Normalized<'tcx,T> {
    pub fn with<U>(self, value: U) -> Normalized<'tcx,U> {
        Normalized { value: value, obligations: self.obligations }
    }
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). If ambiguity arises, which implies that
/// there are unresolved type variables in the projection, we will
/// substitute a fresh type variable `$X` and generate a new
/// obligation `<T as Trait>::Item == $X` for later.
pub fn normalize_projection_type<'a, 'b, 'gcx, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut Vec<PredicateObligation<'tcx>>)
    -> Ty<'tcx>
{
    opt_normalize_projection_type(selcx, param_env, projection_ty.clone(), cause.clone(), depth,
                                  obligations)
        .unwrap_or_else(move || {
            // if we bottom out in ambiguity, create a type variable
            // and a deferred predicate to resolve this when more type
            // information is available.

            let tcx = selcx.infcx().tcx;
            let def_id = projection_ty.item_def_id;
            let ty_var = selcx.infcx().next_ty_var(
                TypeVariableOrigin::NormalizeProjectionType(tcx.def_span(def_id)));
            let projection = ty::Binder::dummy(ty::ProjectionPredicate {
                projection_ty,
                ty: ty_var
            });
            let obligation = Obligation::with_depth(
                cause, depth + 1, param_env, projection.to_predicate());
            obligations.push(obligation);
            ty_var
        })
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). Returns `None` in the case of ambiguity,
/// which indicates that there are unbound type variables.
///
/// This function used to return `Option<NormalizedTy<'tcx>>`, which contains a
/// `Ty<'tcx>` and an obligations vector. But that obligation vector was very
/// often immediately appended to another obligations vector. So now this
/// function takes an obligations vector and appends to it directly, which is
/// slightly uglier but avoids the need for an extra short-lived allocation.
fn opt_normalize_projection_type<'a, 'b, 'gcx, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut Vec<PredicateObligation<'tcx>>)
    -> Option<Ty<'tcx>>
{
    let infcx = selcx.infcx();

    let projection_ty = infcx.resolve_type_vars_if_possible(&projection_ty);
    let cache_key = ProjectionCacheKey { ty: projection_ty };

    debug!("opt_normalize_projection_type(\
           projection_ty={:?}, \
           depth={})",
           projection_ty,
           depth);

    // FIXME(#20304) For now, I am caching here, which is good, but it
    // means we don't capture the type variables that are created in
    // the case of ambiguity. Which means we may create a large stream
    // of such variables. OTOH, if we move the caching up a level, we
    // would not benefit from caching when proving `T: Trait<U=Foo>`
    // bounds. It might be the case that we want two distinct caches,
    // or else another kind of cache entry.

    let cache_result = infcx.projection_cache.borrow_mut().try_start(cache_key);
    match cache_result {
        Ok(()) => { }
        Err(ProjectionCacheEntry::Ambiguous) => {
            // If we found ambiguity the last time, that generally
            // means we will continue to do so until some type in the
            // key changes (and we know it hasn't, because we just
            // fully resolved it). One exception though is closure
            // types, which can transition from having a fixed kind to
            // no kind with no visible change in the key.
            //
            // FIXME(#32286) refactor this so that closure type
            // changes
            debug!("opt_normalize_projection_type: \
                    found cache entry: ambiguous");
            if !projection_ty.has_closure_types() {
                return None;
            }
        }
        Err(ProjectionCacheEntry::InProgress) => {
            // If while normalized A::B, we are asked to normalize
            // A::B, just return A::B itself. This is a conservative
            // answer, in the sense that A::B *is* clearly equivalent
            // to A::B, though there may be a better value we can
            // find.

            // Under lazy normalization, this can arise when
            // bootstrapping.  That is, imagine an environment with a
            // where-clause like `A::B == u32`. Now, if we are asked
            // to normalize `A::B`, we will want to check the
            // where-clauses in scope. So we will try to unify `A::B`
            // with `A::B`, which can trigger a recursive
            // normalization. In that case, I think we will want this code:
            //
            // ```
            // let ty = selcx.tcx().mk_projection(projection_ty.item_def_id,
            //                                    projection_ty.substs;
            // return Some(NormalizedTy { value: v, obligations: vec![] });
            // ```

            debug!("opt_normalize_projection_type: \
                    found cache entry: in-progress");

            // But for now, let's classify this as an overflow:
            let recursion_limit = *selcx.tcx().sess.recursion_limit.get();
            let obligation = Obligation::with_depth(cause,
                                                    recursion_limit,
                                                    param_env,
                                                    projection_ty);
            selcx.infcx().report_overflow_error(&obligation, false);
        }
        Err(ProjectionCacheEntry::NormalizedTy(ty)) => {
            // This is the hottest path in this function.
            //
            // If we find the value in the cache, then return it along
            // with the obligations that went along with it. Note
            // that, when using a fulfillment context, these
            // obligations could in principle be ignored: they have
            // already been registered when the cache entry was
            // created (and hence the new ones will quickly be
            // discarded as duplicated). But when doing trait
            // evaluation this is not the case, and dropping the trait
            // evaluations can causes ICEs (e.g., #43132).
            debug!("opt_normalize_projection_type: \
                    found normalized ty `{:?}`",
                   ty);

            // Once we have inferred everything we need to know, we
            // can ignore the `obligations` from that point on.
            if !infcx.any_unresolved_type_vars(&ty.value) {
                infcx.projection_cache.borrow_mut().complete_normalized(cache_key, &ty);
                // No need to extend `obligations`.
            } else {
                obligations.extend(ty.obligations);
            }

            obligations.push(get_paranoid_cache_value_obligation(infcx,
                                                                 param_env,
                                                                 projection_ty,
                                                                 cause,
                                                                 depth));
            return Some(ty.value);
        }
        Err(ProjectionCacheEntry::Error) => {
            debug!("opt_normalize_projection_type: \
                    found error");
            let result = normalize_to_error(selcx, param_env, projection_ty, cause, depth);
            obligations.extend(result.obligations);
            return Some(result.value)
        }
    }

    let obligation = Obligation::with_depth(cause.clone(), depth, param_env, projection_ty);
    match project_type(selcx, &obligation) {
        Ok(ProjectedTy::Progress(Progress { ty: projected_ty,
                                            obligations: mut projected_obligations })) => {
            // if projection succeeded, then what we get out of this
            // is also non-normalized (consider: it was derived from
            // an impl, where-clause etc) and hence we must
            // re-normalize it

            debug!("opt_normalize_projection_type: \
                    projected_ty={:?} \
                    depth={} \
                    projected_obligations={:?}",
                   projected_ty,
                   depth,
                   projected_obligations);

            let result = if projected_ty.has_projections() {
                let mut normalizer = AssociatedTypeNormalizer::new(selcx,
                                                                   param_env,
                                                                   cause,
                                                                   depth+1);
                let normalized_ty = normalizer.fold(&projected_ty);

                debug!("opt_normalize_projection_type: \
                        normalized_ty={:?} depth={}",
                       normalized_ty,
                       depth);

                projected_obligations.extend(normalizer.obligations);
                Normalized {
                    value: normalized_ty,
                    obligations: projected_obligations,
                }
            } else {
                Normalized {
                    value: projected_ty,
                    obligations: projected_obligations,
                }
            };

            let cache_value = prune_cache_value_obligations(infcx, &result);
            infcx.projection_cache.borrow_mut().insert_ty(cache_key, cache_value);
            obligations.extend(result.obligations);
            Some(result.value)
        }
        Ok(ProjectedTy::NoProgress(projected_ty)) => {
            debug!("opt_normalize_projection_type: \
                    projected_ty={:?} no progress",
                   projected_ty);
            let result = Normalized {
                value: projected_ty,
                obligations: vec![]
            };
            infcx.projection_cache.borrow_mut().insert_ty(cache_key, result.clone());
            // No need to extend `obligations`.
            Some(result.value)
        }
        Err(ProjectionTyError::TooManyCandidates) => {
            debug!("opt_normalize_projection_type: \
                    too many candidates");
            infcx.projection_cache.borrow_mut()
                                  .ambiguous(cache_key);
            None
        }
        Err(ProjectionTyError::TraitSelectionError(_)) => {
            debug!("opt_normalize_projection_type: ERROR");
            // if we got an error processing the `T as Trait` part,
            // just return `ty::err` but add the obligation `T :
            // Trait`, which when processed will cause the error to be
            // reported later

            infcx.projection_cache.borrow_mut()
                                  .error(cache_key);
            let result = normalize_to_error(selcx, param_env, projection_ty, cause, depth);
            obligations.extend(result.obligations);
            Some(result.value)
        }
    }
}

/// If there are unresolved type variables, then we need to include
/// any subobligations that bind them, at least until those type
/// variables are fully resolved.
fn prune_cache_value_obligations<'a, 'gcx, 'tcx>(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
                                                 result: &NormalizedTy<'tcx>)
                                                 -> NormalizedTy<'tcx> {
    if !infcx.any_unresolved_type_vars(&result.value) {
        return NormalizedTy { value: result.value, obligations: vec![] };
    }

    let mut obligations: Vec<_> =
        result.obligations
              .iter()
              .filter(|obligation| match obligation.predicate {
                  // We found a `T: Foo<X = U>` predicate, let's check
                  // if `U` references any unresolved type
                  // variables. In principle, we only care if this
                  // projection can help resolve any of the type
                  // variables found in `result.value` -- but we just
                  // check for any type variables here, for fear of
                  // indirect obligations (e.g., we project to `?0`,
                  // but we have `T: Foo<X = ?1>` and `?1: Bar<X =
                  // ?0>`).
                  ty::Predicate::Projection(ref data) =>
                      infcx.any_unresolved_type_vars(&data.ty()),

                  // We are only interested in `T: Foo<X = U>` predicates, whre
                  // `U` references one of `unresolved_type_vars`. =)
                  _ => false,
              })
              .cloned()
              .collect();

    obligations.shrink_to_fit();

    NormalizedTy { value: result.value, obligations }
}

/// Whenever we give back a cache result for a projection like `<T as
/// Trait>::Item ==> X`, we *always* include the obligation to prove
/// that `T: Trait` (we may also include some other obligations). This
/// may or may not be necessary -- in principle, all the obligations
/// that must be proven to show that `T: Trait` were also returned
/// when the cache was first populated. But there are some vague concerns,
/// and so we take the precautionary measure of including `T: Trait` in
/// the result:
///
/// Concern #1. The current setup is fragile. Perhaps someone could
/// have failed to prove the concerns from when the cache was
/// populated, but also not have used a snapshot, in which case the
/// cache could remain populated even though `T: Trait` has not been
/// shown. In this case, the "other code" is at fault -- when you
/// project something, you are supposed to either have a snapshot or
/// else prove all the resulting obligations -- but it's still easy to
/// get wrong.
///
/// Concern #2. Even within the snapshot, if those original
/// obligations are not yet proven, then we are able to do projections
/// that may yet turn out to be wrong. This *may* lead to some sort
/// of trouble, though we don't have a concrete example of how that
/// can occur yet. But it seems risky at best.
fn get_paranoid_cache_value_obligation<'a, 'gcx, 'tcx>(
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize)
    -> PredicateObligation<'tcx>
{
    let trait_ref = projection_ty.trait_ref(infcx.tcx).to_poly_trait_ref();
    Obligation {
        cause,
        recursion_depth: depth,
        param_env,
        predicate: trait_ref.to_predicate(),
    }
}

/// If we are projecting `<T as Trait>::Item`, but `T: Trait` does not
/// hold. In various error cases, we cannot generate a valid
/// normalized projection. Therefore, we create an inference variable
/// return an associated obligation that, when fulfilled, will lead to
/// an error.
///
/// Note that we used to return `Error` here, but that was quite
/// dubious -- the premise was that an error would *eventually* be
/// reported, when the obligation was processed. But in general once
/// you see a `Error` you are supposed to be able to assume that an
/// error *has been* reported, so that you can take whatever heuristic
/// paths you want to take. To make things worse, it was possible for
/// cycles to arise, where you basically had a setup like `<MyType<$0>
/// as Trait>::Foo == $0`. Here, normalizing `<MyType<$0> as
/// Trait>::Foo> to `[type error]` would lead to an obligation of
/// `<MyType<[type error]> as Trait>::Foo`. We are supposed to report
/// an error for this obligation, but we legitimately should not,
/// because it contains `[type error]`. Yuck! (See issue #29857 for
/// one case where this arose.)
fn normalize_to_error<'a, 'gcx, 'tcx>(selcx: &mut SelectionContext<'a, 'gcx, 'tcx>,
                                      param_env: ty::ParamEnv<'tcx>,
                                      projection_ty: ty::ProjectionTy<'tcx>,
                                      cause: ObligationCause<'tcx>,
                                      depth: usize)
                                      -> NormalizedTy<'tcx>
{
    let trait_ref = projection_ty.trait_ref(selcx.tcx()).to_poly_trait_ref();
    let trait_obligation = Obligation { cause,
                                        recursion_depth: depth,
                                        param_env,
                                        predicate: trait_ref.to_predicate() };
    let tcx = selcx.infcx().tcx;
    let def_id = projection_ty.item_def_id;
    let new_value = selcx.infcx().next_ty_var(
        TypeVariableOrigin::NormalizeProjectionType(tcx.def_span(def_id)));
    Normalized {
        value: new_value,
        obligations: vec![trait_obligation]
    }
}

enum ProjectedTy<'tcx> {
    Progress(Progress<'tcx>),
    NoProgress(Ty<'tcx>),
}

struct Progress<'tcx> {
    ty: Ty<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> Progress<'tcx> {
    fn error<'a,'gcx>(tcx: TyCtxt<'a,'gcx,'tcx>) -> Self {
        Progress {
            ty: tcx.types.err,
            obligations: vec![],
        }
    }

    fn with_addl_obligations(mut self,
                             mut obligations: Vec<PredicateObligation<'tcx>>)
                             -> Self {
        debug!("with_addl_obligations: self.obligations.len={} obligations.len={}",
               self.obligations.len(), obligations.len());

        debug!("with_addl_obligations: self.obligations={:?} obligations={:?}",
               self.obligations, obligations);

        self.obligations.append(&mut obligations);
        self
    }
}

/// Computes the result of a projection type (if we can).
///
/// IMPORTANT:
/// - `obligation` must be fully normalized
fn project_type<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>)
    -> Result<ProjectedTy<'tcx>, ProjectionTyError<'tcx>>
{
    debug!("project(obligation={:?})",
           obligation);

    let recursion_limit = *selcx.tcx().sess.recursion_limit.get();
    if obligation.recursion_depth >= recursion_limit {
        debug!("project: overflow!");
        return Err(ProjectionTyError::TraitSelectionError(SelectionError::Overflow));
    }

    let obligation_trait_ref = &obligation.predicate.trait_ref(selcx.tcx());

    debug!("project: obligation_trait_ref={:?}", obligation_trait_ref);

    if obligation_trait_ref.references_error() {
        return Ok(ProjectedTy::Progress(Progress::error(selcx.tcx())));
    }

    let mut candidates = ProjectionTyCandidateSet::None;

    // Make sure that the following procedures are kept in order. ParamEnv
    // needs to be first because it has highest priority, and Select checks
    // the return value of push_candidate which assumes it's ran at last.
    assemble_candidates_from_param_env(selcx,
                                       obligation,
                                       &obligation_trait_ref,
                                       &mut candidates);

    assemble_candidates_from_trait_def(selcx,
                                       obligation,
                                       &obligation_trait_ref,
                                       &mut candidates);

    assemble_candidates_from_impls(selcx,
                                   obligation,
                                   &obligation_trait_ref,
                                   &mut candidates);

    match candidates {
        ProjectionTyCandidateSet::Single(candidate) => Ok(ProjectedTy::Progress(
            confirm_candidate(selcx,
                              obligation,
                              &obligation_trait_ref,
                              candidate))),
        ProjectionTyCandidateSet::None => Ok(ProjectedTy::NoProgress(
            selcx.tcx().mk_projection(
                obligation.predicate.item_def_id,
                obligation.predicate.substs))),
        // Error occurred while trying to processing impls.
        ProjectionTyCandidateSet::Error(e) => Err(ProjectionTyError::TraitSelectionError(e)),
        // Inherent ambiguity that prevents us from even enumerating the
        // candidates.
        ProjectionTyCandidateSet::Ambiguous => Err(ProjectionTyError::TooManyCandidates),

    }
}

/// The first thing we have to do is scan through the parameter
/// environment to see whether there are any projection predicates
/// there that can answer this question.
fn assemble_candidates_from_param_env<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    debug!("assemble_candidates_from_param_env(..)");
    assemble_candidates_from_predicates(selcx,
                                        obligation,
                                        obligation_trait_ref,
                                        candidate_set,
                                        ProjectionTyCandidate::ParamEnv,
                                        obligation.param_env.caller_bounds.iter().cloned());
}

/// In the case of a nested projection like <<A as Foo>::FooT as Bar>::BarT, we may find
/// that the definition of `Foo` has some clues:
///
/// ```
/// trait Foo {
///     type FooT : Bar<BarT=i32>
/// }
/// ```
///
/// Here, for example, we could conclude that the result is `i32`.
fn assemble_candidates_from_trait_def<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    debug!("assemble_candidates_from_trait_def(..)");

    let tcx = selcx.tcx();
    // Check whether the self-type is itself a projection.
    let (def_id, substs) = match obligation_trait_ref.self_ty().sty {
        ty::Projection(ref data) => {
            (data.trait_ref(tcx).def_id, data.substs)
        }
        ty::Opaque(def_id, substs) => (def_id, substs),
        ty::Infer(ty::TyVar(_)) => {
            // If the self-type is an inference variable, then it MAY wind up
            // being a projected type, so induce an ambiguity.
            candidate_set.mark_ambiguous();
            return;
        }
        _ => return
    };

    // If so, extract what we know from the trait and try to come up with a good answer.
    let trait_predicates = tcx.predicates_of(def_id);
    let bounds = trait_predicates.instantiate(tcx, substs);
    let bounds = elaborate_predicates(tcx, bounds.predicates);
    assemble_candidates_from_predicates(selcx,
                                        obligation,
                                        obligation_trait_ref,
                                        candidate_set,
                                        ProjectionTyCandidate::TraitDef,
                                        bounds)
}

fn assemble_candidates_from_predicates<'cx, 'gcx, 'tcx, I>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
    ctor: fn(ty::PolyProjectionPredicate<'tcx>) -> ProjectionTyCandidate<'tcx>,
    env_predicates: I)
    where I: IntoIterator<Item=ty::Predicate<'tcx>>
{
    debug!("assemble_candidates_from_predicates(obligation={:?})",
           obligation);
    let infcx = selcx.infcx();
    for predicate in env_predicates {
        debug!("assemble_candidates_from_predicates: predicate={:?}",
               predicate);
        if let ty::Predicate::Projection(data) = predicate {
            let same_def_id = data.projection_def_id() == obligation.predicate.item_def_id;

            let is_match = same_def_id && infcx.probe(|_| {
                let data_poly_trait_ref =
                    data.to_poly_trait_ref(infcx.tcx);
                let obligation_poly_trait_ref =
                    obligation_trait_ref.to_poly_trait_ref();
                infcx.at(&obligation.cause, obligation.param_env)
                     .sup(obligation_poly_trait_ref, data_poly_trait_ref)
                     .map(|InferOk { obligations: _, value: () }| {
                         // FIXME(#32730) -- do we need to take obligations
                         // into account in any way? At the moment, no.
                     })
                     .is_ok()
            });

            debug!("assemble_candidates_from_predicates: candidate={:?} \
                    is_match={} same_def_id={}",
                   data, is_match, same_def_id);

            if is_match {
                candidate_set.push_candidate(ctor(data));
            }
        }
    }
}

fn assemble_candidates_from_impls<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    // If we are resolving `<T as TraitRef<...>>::Item == Type`,
    // start out by selecting the predicate `T as TraitRef<...>`:
    let poly_trait_ref = obligation_trait_ref.to_poly_trait_ref();
    let trait_obligation = obligation.with(poly_trait_ref.to_poly_trait_predicate());
    let _ = selcx.infcx().commit_if_ok(|_| {
        let vtable = match selcx.select(&trait_obligation) {
            Ok(Some(vtable)) => vtable,
            Ok(None) => {
                candidate_set.mark_ambiguous();
                return Err(());
            }
            Err(e) => {
                debug!("assemble_candidates_from_impls: selection error {:?}", e);
                candidate_set.mark_error(e);
                return Err(());
            }
        };

        let eligible = match &vtable {
            super::VtableClosure(_) |
            super::VtableGenerator(_) |
            super::VtableFnPointer(_) |
            super::VtableObject(_) |
            super::VtableTraitAlias(_) => {
                debug!("assemble_candidates_from_impls: vtable={:?}",
                       vtable);
                true
            }
            super::VtableImpl(impl_data) => {
                // We have to be careful when projecting out of an
                // impl because of specialization. If we are not in
                // codegen (i.e., projection mode is not "any"), and the
                // impl's type is declared as default, then we disable
                // projection (even if the trait ref is fully
                // monomorphic). In the case where trait ref is not
                // fully monomorphic (i.e., includes type parameters),
                // this is because those type parameters may
                // ultimately be bound to types from other crates that
                // may have specialized impls we can't see. In the
                // case where the trait ref IS fully monomorphic, this
                // is a policy decision that we made in the RFC in
                // order to preserve flexibility for the crate that
                // defined the specializable impl to specialize later
                // for existing types.
                //
                // In either case, we handle this by not adding a
                // candidate for an impl if it contains a `default`
                // type.
                let node_item = assoc_ty_def(selcx,
                                             impl_data.impl_def_id,
                                             obligation.predicate.item_def_id);

                let is_default = if node_item.node.is_from_trait() {
                    // If true, the impl inherited a `type Foo = Bar`
                    // given in the trait, which is implicitly default.
                    // Otherwise, the impl did not specify `type` and
                    // neither did the trait:
                    //
                    // ```rust
                    // trait Foo { type T; }
                    // impl Foo for Bar { }
                    // ```
                    //
                    // This is an error, but it will be
                    // reported in `check_impl_items_against_trait`.
                    // We accept it here but will flag it as
                    // an error when we confirm the candidate
                    // (which will ultimately lead to `normalize_to_error`
                    // being invoked).
                    node_item.item.defaultness.has_value()
                } else {
                    node_item.item.defaultness.is_default() ||
                        selcx.tcx().impl_is_default(node_item.node.def_id())
                };

                // Only reveal a specializable default if we're past type-checking
                // and the obligations is monomorphic, otherwise passes such as
                // transmute checking and polymorphic MIR optimizations could
                // get a result which isn't correct for all monomorphizations.
                if !is_default {
                    true
                } else if obligation.param_env.reveal == Reveal::All {
                    debug_assert!(!poly_trait_ref.needs_infer());
                    if !poly_trait_ref.needs_subst() {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            super::VtableParam(..) => {
                // This case tell us nothing about the value of an
                // associated type. Consider:
                //
                // ```
                // trait SomeTrait { type Foo; }
                // fn foo<T:SomeTrait>(...) { }
                // ```
                //
                // If the user writes `<T as SomeTrait>::Foo`, then the `T
                // : SomeTrait` binding does not help us decide what the
                // type `Foo` is (at least, not more specifically than
                // what we already knew).
                //
                // But wait, you say! What about an example like this:
                //
                // ```
                // fn bar<T:SomeTrait<Foo=usize>>(...) { ... }
                // ```
                //
                // Doesn't the `T : Sometrait<Foo=usize>` predicate help
                // resolve `T::Foo`? And of course it does, but in fact
                // that single predicate is desugared into two predicates
                // in the compiler: a trait predicate (`T : SomeTrait`) and a
                // projection. And the projection where clause is handled
                // in `assemble_candidates_from_param_env`.
                false
            }
            super::VtableAutoImpl(..) |
            super::VtableBuiltin(..) => {
                // These traits have no associated types.
                span_bug!(
                    obligation.cause.span,
                    "Cannot project an associated type from `{:?}`",
                    vtable);
            }
        };

        if eligible {
            if candidate_set.push_candidate(ProjectionTyCandidate::Select(vtable)) {
                Ok(())
            } else {
                Err(())
            }
        } else {
            Err(())
        }
    });
}

fn confirm_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate: ProjectionTyCandidate<'tcx>)
    -> Progress<'tcx>
{
    debug!("confirm_candidate(candidate={:?}, obligation={:?})",
           candidate,
           obligation);

    match candidate {
        ProjectionTyCandidate::ParamEnv(poly_projection) |
        ProjectionTyCandidate::TraitDef(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection)
        }

        ProjectionTyCandidate::Select(vtable) => {
            confirm_select_candidate(selcx, obligation, obligation_trait_ref, vtable)
        }
    }
}

fn confirm_select_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    vtable: Selection<'tcx>)
    -> Progress<'tcx>
{
    match vtable {
        super::VtableImpl(data) =>
            confirm_impl_candidate(selcx, obligation, data),
        super::VtableGenerator(data) =>
            confirm_generator_candidate(selcx, obligation, data),
        super::VtableClosure(data) =>
            confirm_closure_candidate(selcx, obligation, data),
        super::VtableFnPointer(data) =>
            confirm_fn_pointer_candidate(selcx, obligation, data),
        super::VtableObject(_) =>
            confirm_object_candidate(selcx, obligation, obligation_trait_ref),
        super::VtableAutoImpl(..) |
        super::VtableParam(..) |
        super::VtableBuiltin(..) |
        super::VtableTraitAlias(..) =>
            // we don't create Select candidates with this kind of resolution
            span_bug!(
                obligation.cause.span,
                "Cannot project an associated type from `{:?}`",
                vtable),
    }
}

fn confirm_object_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation:  &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>)
    -> Progress<'tcx>
{
    let self_ty = obligation_trait_ref.self_ty();
    let object_ty = selcx.infcx().shallow_resolve(self_ty);
    debug!("confirm_object_candidate(object_ty={:?})",
           object_ty);
    let data = match object_ty.sty {
        ty::Dynamic(ref data, ..) => data,
        _ => {
            span_bug!(
                obligation.cause.span,
                "confirm_object_candidate called with non-object: {:?}",
                object_ty)
        }
    };
    let env_predicates = data.projection_bounds().map(|p| {
        p.with_self_ty(selcx.tcx(), object_ty).to_predicate()
    }).collect();
    let env_predicate = {
        let env_predicates = elaborate_predicates(selcx.tcx(), env_predicates);

        // select only those projections that are actually projecting an
        // item with the correct name
        let env_predicates = env_predicates.filter_map(|p| match p {
            ty::Predicate::Projection(data) =>
                if data.projection_def_id() == obligation.predicate.item_def_id {
                    Some(data)
                } else {
                    None
                },
            _ => None
        });

        // select those with a relevant trait-ref
        let mut env_predicates = env_predicates.filter(|data| {
            let data_poly_trait_ref = data.to_poly_trait_ref(selcx.tcx());
            let obligation_poly_trait_ref = obligation_trait_ref.to_poly_trait_ref();
            selcx.infcx().probe(|_|
                selcx.infcx().at(&obligation.cause, obligation.param_env)
                             .sup(obligation_poly_trait_ref, data_poly_trait_ref)
                             .is_ok()
            )
        });

        // select the first matching one; there really ought to be one or
        // else the object type is not WF, since an object type should
        // include all of its projections explicitly
        match env_predicates.next() {
            Some(env_predicate) => env_predicate,
            None => {
                debug!("confirm_object_candidate: no env-predicate \
                        found in object type `{:?}`; ill-formed",
                       object_ty);
                return Progress::error(selcx.tcx());
            }
        }
    };

    confirm_param_env_candidate(selcx, obligation, env_predicate)
}

fn confirm_generator_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    vtable: VtableGeneratorData<'tcx, PredicateObligation<'tcx>>)
    -> Progress<'tcx>
{
    let gen_sig = vtable.substs.poly_sig(vtable.generator_def_id, selcx.tcx());
    let Normalized {
        value: gen_sig,
        obligations
    } = normalize_with_depth(selcx,
                             obligation.param_env,
                             obligation.cause.clone(),
                             obligation.recursion_depth+1,
                             &gen_sig);

    debug!("confirm_generator_candidate: obligation={:?},gen_sig={:?},obligations={:?}",
           obligation,
           gen_sig,
           obligations);

    let tcx = selcx.tcx();

    let gen_def_id = tcx.lang_items().gen_trait().unwrap();

    let predicate =
        tcx.generator_trait_ref_and_outputs(gen_def_id,
                                            obligation.predicate.self_ty(),
                                            gen_sig)
        .map_bound(|(trait_ref, yield_ty, return_ty)| {
            let name = tcx.associated_item(obligation.predicate.item_def_id).ident.name;
            let ty = if name == "Return" {
                return_ty
            } else if name == "Yield" {
                yield_ty
            } else {
                bug!()
            };

            ty::ProjectionPredicate {
                projection_ty: ty::ProjectionTy {
                    substs: trait_ref.substs,
                    item_def_id: obligation.predicate.item_def_id,
                },
                ty: ty
            }
        });

    confirm_param_env_candidate(selcx, obligation, predicate)
        .with_addl_obligations(vtable.nested)
        .with_addl_obligations(obligations)
}

fn confirm_fn_pointer_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_pointer_vtable: VtableFnPointerData<'tcx, PredicateObligation<'tcx>>)
    -> Progress<'tcx>
{
    let fn_type = selcx.infcx().shallow_resolve(fn_pointer_vtable.fn_ty);
    let sig = fn_type.fn_sig(selcx.tcx());
    let Normalized {
        value: sig,
        obligations
    } = normalize_with_depth(selcx,
                             obligation.param_env,
                             obligation.cause.clone(),
                             obligation.recursion_depth+1,
                             &sig);

    confirm_callable_candidate(selcx, obligation, sig, util::TupleArgumentsFlag::Yes)
        .with_addl_obligations(fn_pointer_vtable.nested)
        .with_addl_obligations(obligations)
}

fn confirm_closure_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    vtable: VtableClosureData<'tcx, PredicateObligation<'tcx>>)
    -> Progress<'tcx>
{
    let tcx = selcx.tcx();
    let infcx = selcx.infcx();
    let closure_sig_ty = vtable.substs.closure_sig_ty(vtable.closure_def_id, tcx);
    let closure_sig = infcx.shallow_resolve(&closure_sig_ty).fn_sig(tcx);
    let Normalized {
        value: closure_sig,
        obligations
    } = normalize_with_depth(selcx,
                             obligation.param_env,
                             obligation.cause.clone(),
                             obligation.recursion_depth+1,
                             &closure_sig);

    debug!("confirm_closure_candidate: obligation={:?},closure_sig={:?},obligations={:?}",
           obligation,
           closure_sig,
           obligations);

    confirm_callable_candidate(selcx,
                               obligation,
                               closure_sig,
                               util::TupleArgumentsFlag::No)
        .with_addl_obligations(vtable.nested)
        .with_addl_obligations(obligations)
}

fn confirm_callable_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
    flag: util::TupleArgumentsFlag)
    -> Progress<'tcx>
{
    let tcx = selcx.tcx();

    debug!("confirm_callable_candidate({:?},{:?})",
           obligation,
           fn_sig);

    // the `Output` associated type is declared on `FnOnce`
    let fn_once_def_id = tcx.lang_items().fn_once_trait().unwrap();

    let predicate =
        tcx.closure_trait_ref_and_return_type(fn_once_def_id,
                                              obligation.predicate.self_ty(),
                                              fn_sig,
                                              flag)
        .map_bound(|(trait_ref, ret_type)|
            ty::ProjectionPredicate {
                projection_ty: ty::ProjectionTy::from_ref_and_name(
                    tcx,
                    trait_ref,
                    Ident::from_str(FN_OUTPUT_NAME),
                ),
                ty: ret_type
            }
        );

    confirm_param_env_candidate(selcx, obligation, predicate)
}

fn confirm_param_env_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    poly_cache_entry: ty::PolyProjectionPredicate<'tcx>,
) -> Progress<'tcx> {
    let infcx = selcx.infcx();
    let cause = &obligation.cause;
    let param_env = obligation.param_env;

    let (cache_entry, _) =
        infcx.replace_bound_vars_with_fresh_vars(
            cause.span,
            LateBoundRegionConversionTime::HigherRankedType,
            &poly_cache_entry);

    let cache_trait_ref = cache_entry.projection_ty.trait_ref(infcx.tcx);
    let obligation_trait_ref = obligation.predicate.trait_ref(infcx.tcx);
    match infcx.at(cause, param_env).eq(cache_trait_ref, obligation_trait_ref) {
        Ok(InferOk { value: _, obligations }) => {
            Progress {
                ty: cache_entry.ty,
                obligations,
            }
        }
        Err(e) => {
            span_bug!(
                obligation.cause.span,
                "Failed to unify obligation `{:?}` \
                 with poly_projection `{:?}`: {:?}",
                obligation,
                poly_cache_entry,
                e);
        }
    }
}

fn confirm_impl_candidate<'cx, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    impl_vtable: VtableImplData<'tcx, PredicateObligation<'tcx>>)
    -> Progress<'tcx>
{
    let VtableImplData { impl_def_id, substs, nested } = impl_vtable;

    let tcx = selcx.tcx();
    let param_env = obligation.param_env;
    let assoc_ty = assoc_ty_def(selcx, impl_def_id, obligation.predicate.item_def_id);

    if !assoc_ty.item.defaultness.has_value() {
        // This means that the impl is missing a definition for the
        // associated type. This error will be reported by the type
        // checker method `check_impl_items_against_trait`, so here we
        // just return Error.
        debug!("confirm_impl_candidate: no associated type {:?} for {:?}",
               assoc_ty.item.ident,
               obligation.predicate);
        return Progress {
            ty: tcx.types.err,
            obligations: nested,
        };
    }
    let substs = translate_substs(selcx.infcx(), param_env, impl_def_id, substs, assoc_ty.node);
    let ty = if let ty::AssociatedKind::Existential = assoc_ty.item.kind {
        let item_substs = InternalSubsts::identity_for_item(tcx, assoc_ty.item.def_id);
        tcx.mk_opaque(assoc_ty.item.def_id, item_substs)
    } else {
        tcx.type_of(assoc_ty.item.def_id)
    };
    Progress {
        ty: ty.subst(tcx, substs),
        obligations: nested,
    }
}

/// Locate the definition of an associated type in the specialization hierarchy,
/// starting from the given impl.
///
/// Based on the "projection mode", this lookup may in fact only examine the
/// topmost impl. See the comments for `Reveal` for more details.
fn assoc_ty_def<'cx, 'gcx, 'tcx>(
    selcx: &SelectionContext<'cx, 'gcx, 'tcx>,
    impl_def_id: DefId,
    assoc_ty_def_id: DefId)
    -> specialization_graph::NodeItem<ty::AssociatedItem>
{
    let tcx = selcx.tcx();
    let assoc_ty_name = tcx.associated_item(assoc_ty_def_id).ident;
    let trait_def_id = tcx.impl_trait_ref(impl_def_id).unwrap().def_id;
    let trait_def = tcx.trait_def(trait_def_id);

    // This function may be called while we are still building the
    // specialization graph that is queried below (via TraidDef::ancestors()),
    // so, in order to avoid unnecessary infinite recursion, we manually look
    // for the associated item at the given impl.
    // If there is no such item in that impl, this function will fail with a
    // cycle error if the specialization graph is currently being built.
    let impl_node = specialization_graph::Node::Impl(impl_def_id);
    for item in impl_node.items(tcx) {
        if item.kind == ty::AssociatedKind::Type &&
                tcx.hygienic_eq(item.ident, assoc_ty_name, trait_def_id) {
            return specialization_graph::NodeItem {
                node: specialization_graph::Node::Impl(impl_def_id),
                item,
            };
        }
    }

    if let Some(assoc_item) = trait_def
        .ancestors(tcx, impl_def_id)
        .defs(tcx, assoc_ty_name, ty::AssociatedKind::Type, trait_def_id)
        .next() {
        assoc_item
    } else {
        // This is saying that neither the trait nor
        // the impl contain a definition for this
        // associated type.  Normally this situation
        // could only arise through a compiler bug --
        // if the user wrote a bad item name, it
        // should have failed in astconv.
        bug!("No associated type `{}` for {}",
             assoc_ty_name,
             tcx.item_path_str(impl_def_id))
    }
}

// # Cache

/// The projection cache. Unlike the standard caches, this can include
/// infcx-dependent type variables, therefore we have to roll the
/// cache back each time we roll a snapshot back, to avoid assumptions
/// on yet-unresolved inference variables. Types with placeholder
/// regions also have to be removed when the respective snapshot ends.
///
/// Because of that, projection cache entries can be "stranded" and left
/// inaccessible when type variables inside the key are resolved. We make no
/// attempt to recover or remove "stranded" entries, but rather let them be
/// (for the lifetime of the infcx).
///
/// Entries in the projection cache might contain inference variables
/// that will be resolved by obligations on the projection cache entry (e.g.,
/// when a type parameter in the associated type is constrained through
/// an "RFC 447" projection on the impl).
///
/// When working with a fulfillment context, the derived obligations of each
/// projection cache entry will be registered on the fulfillcx, so any users
/// that can wait for a fulfillcx fixed point need not care about this. However,
/// users that don't wait for a fixed point (e.g., trait evaluation) have to
/// resolve the obligations themselves to make sure the projected result is
/// ok and avoid issues like #43132.
///
/// If that is done, after evaluation the obligations, it is a good idea to
/// call `ProjectionCache::complete` to make sure the obligations won't be
/// re-evaluated and avoid an exponential worst-case.
//
// FIXME: we probably also want some sort of cross-infcx cache here to
// reduce the amount of duplication. Let's see what we get with the Chalk reforms.
#[derive(Default)]
pub struct ProjectionCache<'tcx> {
    map: SnapshotMap<ProjectionCacheKey<'tcx>, ProjectionCacheEntry<'tcx>>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProjectionCacheKey<'tcx> {
    ty: ty::ProjectionTy<'tcx>
}

impl<'cx, 'gcx, 'tcx> ProjectionCacheKey<'tcx> {
    pub fn from_poly_projection_predicate(selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
                                          predicate: &ty::PolyProjectionPredicate<'tcx>)
                                          -> Option<Self>
    {
        let infcx = selcx.infcx();
        // We don't do cross-snapshot caching of obligations with escaping regions,
        // so there's no cache key to use
        predicate.no_bound_vars()
            .map(|predicate| ProjectionCacheKey {
                // We don't attempt to match up with a specific type-variable state
                // from a specific call to `opt_normalize_projection_type` - if
                // there's no precise match, the original cache entry is "stranded"
                // anyway.
                ty: infcx.resolve_type_vars_if_possible(&predicate.projection_ty)
            })
    }
}

#[derive(Clone, Debug)]
enum ProjectionCacheEntry<'tcx> {
    InProgress,
    Ambiguous,
    Error,
    NormalizedTy(NormalizedTy<'tcx>),
}

// N.B., intentionally not Clone
pub struct ProjectionCacheSnapshot {
    snapshot: Snapshot,
}

impl<'tcx> ProjectionCache<'tcx> {
    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn snapshot(&mut self) -> ProjectionCacheSnapshot {
        ProjectionCacheSnapshot { snapshot: self.map.snapshot() }
    }

    pub fn rollback_to(&mut self, snapshot: ProjectionCacheSnapshot) {
        self.map.rollback_to(snapshot.snapshot);
    }

    pub fn rollback_placeholder(&mut self, snapshot: &ProjectionCacheSnapshot) {
        self.map.partial_rollback(&snapshot.snapshot, &|k| k.ty.has_re_placeholders());
    }

    pub fn commit(&mut self, snapshot: ProjectionCacheSnapshot) {
        self.map.commit(snapshot.snapshot);
    }

    /// Try to start normalize `key`; returns an error if
    /// normalization already occurred (this error corresponds to a
    /// cache hit, so it's actually a good thing).
    fn try_start(&mut self, key: ProjectionCacheKey<'tcx>)
                 -> Result<(), ProjectionCacheEntry<'tcx>> {
        if let Some(entry) = self.map.get(&key) {
            return Err(entry.clone());
        }

        self.map.insert(key, ProjectionCacheEntry::InProgress);
        Ok(())
    }

    /// Indicates that `key` was normalized to `value`.
    fn insert_ty(&mut self, key: ProjectionCacheKey<'tcx>, value: NormalizedTy<'tcx>) {
        debug!("ProjectionCacheEntry::insert_ty: adding cache entry: key={:?}, value={:?}",
               key, value);
        let fresh_key = self.map.insert(key, ProjectionCacheEntry::NormalizedTy(value));
        assert!(!fresh_key, "never started projecting `{:?}`", key);
    }

    /// Mark the relevant projection cache key as having its derived obligations
    /// complete, so they won't have to be re-computed (this is OK to do in a
    /// snapshot - if the snapshot is rolled back, the obligations will be
    /// marked as incomplete again).
    pub fn complete(&mut self, key: ProjectionCacheKey<'tcx>) {
        let ty = match self.map.get(&key) {
            Some(&ProjectionCacheEntry::NormalizedTy(ref ty)) => {
                debug!("ProjectionCacheEntry::complete({:?}) - completing {:?}",
                       key, ty);
                ty.value
            }
            ref value => {
                // Type inference could "strand behind" old cache entries. Leave
                // them alone for now.
                debug!("ProjectionCacheEntry::complete({:?}) - ignoring {:?}",
                       key, value);
                return
            }
        };

        self.map.insert(key, ProjectionCacheEntry::NormalizedTy(Normalized {
            value: ty,
            obligations: vec![]
        }));
    }

    /// A specialized version of `complete` for when the key's value is known
    /// to be a NormalizedTy.
    pub fn complete_normalized(&mut self, key: ProjectionCacheKey<'tcx>, ty: &NormalizedTy<'tcx>) {
        // We want to insert `ty` with no obligations. If the existing value
        // already has no obligations (as is common) we don't insert anything.
        if !ty.obligations.is_empty() {
            self.map.insert(key, ProjectionCacheEntry::NormalizedTy(Normalized {
                value: ty.value,
                obligations: vec![]
            }));
        }
    }

    /// Indicates that trying to normalize `key` resulted in
    /// ambiguity. No point in trying it again then until we gain more
    /// type information (in which case, the "fully resolved" key will
    /// be different).
    fn ambiguous(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map.insert(key, ProjectionCacheEntry::Ambiguous);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }

    /// Indicates that trying to normalize `key` resulted in
    /// error.
    fn error(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map.insert(key, ProjectionCacheEntry::Error);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }
}
