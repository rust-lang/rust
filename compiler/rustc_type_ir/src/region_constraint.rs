//! The bulk of the logic for implementing `-Zassumptions-on-binders`

use derive_where::derive_where;
use indexmap::IndexSet;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
#[cfg(feature = "nightly")]
use rustc_data_structures::transitive_relation::{TransitiveRelation, TransitiveRelationBuilder};
use tracing::{debug, instrument};

// Workaround for TransitiveRelation being in rustc_data_structures which isn't accessible on stable
#[cfg(not(feature = "nightly"))]
#[derive(Default, Clone, Debug)]
pub struct TransitiveRelation<T>(T);
#[cfg(not(feature = "nightly"))]
impl<T> TransitiveRelation<T> {
    pub fn reachable_from(&self, _data: T) -> Vec<T> {
        unreachable!("-Zassumptions-on-binders is not supported for r-a")
    }

    pub fn base_edges(&self) -> impl Iterator<Item = (T, T)> {
        unreachable!("-Zassumptions-on-binders is not supported for r-a");

        #[allow(unreachable_code)]
        [].into_iter()
    }
}
#[derive(Clone, Debug)]
#[cfg(not(feature = "nightly"))]
pub struct TransitiveRelationBuilder<T>(T);
#[cfg(not(feature = "nightly"))]
impl<T> TransitiveRelationBuilder<T> {
    pub fn freeze(self) -> TransitiveRelation<T> {
        unreachable!("-Zassumptions-on-binders is not supported for r-a")
    }

    pub fn add(&mut self, _: T, _: T) {
        unreachable!("-Zassumptions-on-binders is not supported for r-a")
    }
}
#[cfg(not(feature = "nightly"))]
impl<T> Default for TransitiveRelationBuilder<T> {
    fn default() -> Self {
        unreachable!("-Zassumptions-on-binders is not supported for r-a")
    }
}

use crate::data_structures::IndexMap;
use crate::fold::TypeSuperFoldable;
use crate::inherent::*;
use crate::relate::{Relate, RelateResult, TypeRelation, VarianceDiagInfo};
use crate::{
    AliasTy, Binder, BoundRegion, BoundVar, BoundVariableKind, DebruijnIndex, FallibleTypeFolder,
    GenericTypeVisitable, InferCtxtLike, Interner, IsRigid, OutlivesClause, Region, RegionKind,
    TyKind, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor, TypingMode, UniverseIndex,
    Variance, VisitorResult, max_universe, set_aliases_to_non_rigid, try_visit,
    walk_visitable_list,
};

#[derive_where(Clone, Debug; I: Interner)]
pub struct Assumptions<I: Interner> {
    pub type_outlives: Vec<Binder<I, OutlivesClause<I, I::Ty>>>,
    pub region_outlives: TransitiveRelation<Region<I>>,
    pub inverse_region_outlives: TransitiveRelation<Region<I>>,
}

impl<I: Interner> Assumptions<I> {
    pub fn empty() -> Self {
        Self {
            type_outlives: Vec::new(),
            region_outlives: TransitiveRelationBuilder::default().freeze(),
            inverse_region_outlives: TransitiveRelationBuilder::default().freeze(),
        }
    }

    pub fn new(
        type_outlives: Vec<Binder<I, OutlivesClause<I, I::Ty>>>,
        region_outlives: TransitiveRelation<Region<I>>,
    ) -> Self {
        Self {
            inverse_region_outlives: {
                let mut builder = TransitiveRelationBuilder::default();
                for (r1, r2) in region_outlives.base_edges() {
                    builder.add(r2, r1);
                }
                builder.freeze()
            },
            type_outlives,
            region_outlives,
        }
    }
}

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, S)]
#[derive(GenericTypeVisitable)]
pub enum RegionConstraint<I: Interner, S = ()> {
    Ambiguity(S),
    RegionOutlives(Region<I>, Region<I>, S),
    /// Requirement that a (potentially higher ranked) alias outlives some (potentially higher ranked)
    /// region due to an assumption in the environment. This cannot be satisfied via component outlives
    /// or item bounds.
    ///
    /// We cannot eagerly look at assumptions as we are usually working with an incomplete set of assumptions
    /// and there may wind up being assumptions we can use to prove this when we're in a smaller universe.
    ///
    /// We eagerly destructure alias outlives requirements into region outlives requirements corresponding to
    /// component outlives & item bound outlives rules, leaving only param env candidates.
    AliasTyOutlivesViaEnv(Binder<I, (AliasTy<I>, Region<I>)>, S),
    /// This is an `I::Ty` for two reasons:
    /// 1. We need the type visitable impl to be able to `visit_ty` on this so canonicalization
    ///    knows about the placeholder
    /// 2. When exiting the trait solver there may be placeholder outlives corresponding to params
    ///    from the root universe. These need to be changed from a `Placeholder` to the original
    ///    `Param`.
    ///
    /// We cannot eagerly look at assumptions as we are usually working with an incomplete set of assumptions
    /// and there may wind up being assumptions we can use to prove this when we're in a smaller universe.
    PlaceholderTyOutlives(I::Ty, Region<I>, S),

    And(#[generic_type_visitable(bounds())] Box<[RegionConstraint<I, S>]>),
    Or(#[generic_type_visitable(bounds())] Box<[RegionConstraint<I, S>]>),
}

impl<I: Interner> Default for RegionConstraint<I> {
    fn default() -> Self {
        Self::new_true()
    }
}

impl<I: Interner, S: Clone + std::fmt::Debug> RegionConstraint<I, S> {
    pub fn new_true() -> Self {
        RegionConstraint::And(Box::new([]))
    }

    pub fn is_true(&self) -> bool {
        match self {
            Self::And(and) => and.is_empty(),
            _ => false,
        }
    }

    pub fn new_false() -> Self {
        RegionConstraint::Or(Box::new([]))
    }

    pub fn is_false(&self) -> bool {
        match self {
            Self::Or(or) => or.is_empty(),
            _ => false,
        }
    }

    pub fn is_or(&self) -> bool {
        matches!(self, Self::Or(_))
    }

    pub fn unwrap_or(self) -> Box<[RegionConstraint<I, S>]> {
        match self {
            Self::Or(ors) => ors,
            _ => panic!("`unwrap_or` on non-Or: {self:?}"),
        }
    }

    pub fn unwrap_and(self) -> Box<[RegionConstraint<I, S>]> {
        match self {
            Self::And(ands) => ands,
            _ => panic!("`unwrap_and` on non-And: {self:?}"),
        }
    }

    pub fn is_and(&self) -> bool {
        matches!(self, Self::And(_))
    }

    pub fn is_ambig(&self) -> bool {
        matches!(self, Self::Ambiguity(_))
    }

    pub fn and(self, other: RegionConstraint<I, S>) -> RegionConstraint<I, S> {
        use RegionConstraint::*;

        match (self, other) {
            (And(a_ands), And(b_ands)) => And(a_ands
                .into_iter()
                .chain(b_ands.into_iter())
                .collect::<Vec<_>>()
                .into_boxed_slice()),
            (And(ands), other) | (other, And(ands)) => {
                And(ands.into_iter().chain([other]).collect::<Vec<_>>().into_boxed_slice())
            }
            (this, other) => And(Box::new([this, other])),
        }
    }
}

/// Takes any constraints involving placeholders from the current universe and eagerly checks them.
/// This can be done a few ways:
/// - There's an assumption on the binder introducing the placeholder which means the constraint is satisfied (true)
/// - There's assumptions on the binder introducing the placeholder which allow us to rewrite the constraint in
///    terms of lower universe variables. For example given `for<'a> where('b: 'a) { prove(T: '!a_u1) }` we can
///    convert this constraint to `T: 'b` which no longer references anything from `u1`.
/// - There are no relevant assumptions so we can neither rewrite the constraint nor consider it satisfied (false)
/// - We failed to compute the full set of assumptions when entering the binder corresponding to `u`. (ambiguity)
///
/// After handling all of the region constraints in `u` we then evaluate the entire constraint as much as possible,
/// propagating true/false/ambiguity as close to the root of the constraint as we can. The returned constraint should
/// be checked for whether it is true/false/ambiguous as that should affect the result of whatever operation required
/// entering the binder corresponding to `u`.
#[instrument(level = "debug", skip(infcx), ret)]
pub fn eagerly_handle_placeholders_in_universe<Infcx: InferCtxtLike<Interner = I>, I: Interner>(
    infcx: &Infcx,
    constraint: RegionConstraint<I>,
    u: UniverseIndex,
) -> RegionConstraint<I> {
    let assumptions = infcx.get_placeholder_assumptions(u);

    // 1. rewrite type outlives constraints involving things from `u` into either region constraints
    //     involving things from `u` or type outlives constraints not involving things from `u`
    //
    //    IOW, we only want to encounter things from `u` as part of region out lives constraints.
    let constraint = rewrite_type_outlives_constraints_in_universe_for_eager_placeholder_handling(
        infcx,
        constraint.canonical_form(),
        u,
        &assumptions,
    );

    // 2. compute transitive region outlives and get a new set of region outlives constraints by
    //     looking for every region which either a placeholder_u flows into it, or it flows into
    //     the placeholder.
    let constraint = compute_new_region_constraints(infcx, constraint.canonical_form(), u);

    // 3. rewrite region outlives constraints (potentially to false/true)
    let constraint = pull_region_outlives_constraints_out_of_universe(
        infcx,
        constraint.canonical_form(),
        u,
        &assumptions,
    );

    // 4. actually evaluate the constraint to eagerly error on false
    evaluate_solver_constraint(&constraint.canonical_form())
}

/// Filter our region constraints to not include constraints between region variables from `u` and
/// other regions as those are always satisfied. This requires some care to handle correctly for example:
/// `'!a_u1: '?x_u1: '!b_u1` should result in us requiring `'!a_u1: '!b_u1` rather than dropping the two
/// constraints entirely.
///
/// The only constraints involving things from `u` should be region outlives constraints at this point. Type
/// outlives constraints should have been handled already either by destructuring into region outlives or by
/// being rewritten in terms of smaller universe variables.
#[instrument(level = "debug", skip(infcx), ret)]
fn compute_new_region_constraints<Infcx: InferCtxtLike<Interner = I>, I: Interner>(
    infcx: &Infcx,
    constraint: RegionConstraint<I>,
    u: UniverseIndex,
) -> RegionConstraint<I> {
    use RegionConstraint::*;

    let extend_from_and = |builder: &mut TransitiveRelationBuilder<_>,
                           regions: &mut IndexSet<_>,
                           constraints: &mut Vec<_>,
                           and: RegionConstraint<I>| {
        for c in and.unwrap_and() {
            match c {
                Ambiguity(()) | PlaceholderTyOutlives(..) | AliasTyOutlivesViaEnv(..) => {
                    constraints.push(c.clone())
                }
                RegionOutlives(r1, r2, ()) => {
                    regions.insert(r1);
                    regions.insert(r2);
                    builder.add(r2, r1);
                }
                Or(_) | And(_) => unreachable!(),
            }
        }
    };

    let mut new_ands = Vec::new();
    for and in constraint.unwrap_or() {
        let mut region_flows_builder = TransitiveRelationBuilder::default();
        let mut regions = IndexSet::new();
        let mut constraints = Vec::new();

        extend_from_and(&mut region_flows_builder, &mut regions, &mut constraints, and);

        let region_flow = region_flows_builder.freeze();
        for r in regions.into_iter() {
            for ub in region_flow.reachable_from(r) {
                // we want to retain any region constraints between two "placeholder-likes" where for our
                // purposes a placeholder-like is either a placeholder or variable in a lower universe
                let is_placeholder_like = |r: Region<I>| match r.kind() {
                    RegionKind::ReLateParam(..)
                    | RegionKind::ReEarlyParam(..)
                    | RegionKind::RePlaceholder(..)
                    | RegionKind::ReStatic => true,
                    RegionKind::ReVar(..) => max_universe(infcx, r) < u,
                    RegionKind::ReError(..) => false,
                    RegionKind::ReErased | RegionKind::ReBound(..) => unreachable!(),
                };

                if is_placeholder_like(r) && is_placeholder_like(ub) {
                    constraints.push(RegionOutlives(ub, r, ()));
                }
            }
        }

        new_ands.push(And(constraints.into_boxed_slice()))
    }

    Or(new_ands.into_boxed_slice())
}

/// Evaluate ANDs and ORs to true/false/ambiguous based on whether their arguments are true/false/ambiguous
#[instrument(level = "debug", ret)]
pub fn evaluate_solver_constraint<I: Interner, S: Clone + std::fmt::Debug>(
    constraint: &RegionConstraint<I, S>,
) -> RegionConstraint<I, S> {
    use RegionConstraint::*;
    match constraint {
        Ambiguity(_)
        | RegionOutlives(..)
        | AliasTyOutlivesViaEnv(..)
        | PlaceholderTyOutlives(..) => constraint.clone(),
        And(and) => {
            let mut and_constraints = Vec::new();
            let mut ambiguity = None;
            for c in and.iter() {
                let evaluated_constraint = evaluate_solver_constraint(c);
                if evaluated_constraint.is_true() {
                    // - do nothing
                } else if evaluated_constraint.is_false() {
                    return RegionConstraint::new_false();
                } else if let Ambiguity(span) = evaluated_constraint {
                    ambiguity.get_or_insert(span);
                } else {
                    and_constraints.push(evaluated_constraint);
                }
            }

            ambiguity.map_or_else(
                || RegionConstraint::And(and_constraints.into_boxed_slice()),
                RegionConstraint::Ambiguity,
            )
        }
        Or(or) => {
            let mut or_constraints = Vec::new();
            let mut ambiguity = None;
            for c in or.iter() {
                let evaluated_constraint = evaluate_solver_constraint(c);
                if evaluated_constraint.is_false() {
                    // do nothing
                } else if evaluated_constraint.is_true() {
                    return RegionConstraint::new_true();
                } else if let Ambiguity(span) = evaluated_constraint {
                    ambiguity.get_or_insert(span);
                } else {
                    or_constraints.push(evaluated_constraint);
                }
            }

            ambiguity.map_or_else(
                || RegionConstraint::Or(or_constraints.into_boxed_slice()),
                RegionConstraint::Ambiguity,
            )
        }
    }
}

/// Handles converting region outlives constraints involving placeholders from `u` into OR constraints
/// involving regions from smaller universes with known relationships to the placeholder. For example:
/// ```ignore (not rust)
/// for<'a, 'b> where(
///     'c: 'b, 'd: 'b,
///     'a: 'e, 'a: 'f,
/// ) {
///     'a_u1: 'b_u1
/// }
/// ```
/// will get converted to:
/// ```ignore (not rust)
/// OR(
///     'e: 'c,
///     'e: 'd,
///     'f: 'c,
///     'f: 'd,
/// )
/// ```
/// if we are handling constraints in `u1`.
#[instrument(level = "debug", skip(infcx), ret)]
fn pull_region_outlives_constraints_out_of_universe<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    constraint: RegionConstraint<I>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> RegionConstraint<I> {
    assert!(max_universe(infcx, constraint.clone()) <= u);

    // FIXME(-Zassumptions-on-binders): we don't lower universes of region variables when exiting `u`
    // this seems dubious/potentially wrong? we can't just blindly do this though as if we had something
    // like `!T_u -> ?x_u -> !U_u` then lowering `?x` to `u-1` when exiting `u` would be wrong.
    //
    // I'm not even sure this would be necessary given we filter out region constraints involving regions#
    // from the current universe and only retain those between placeholders.

    use RegionConstraint::*;
    let pull_and = |and: RegionConstraint<I>| {
        let mut pulled_constraints = Vec::new();
        for c in and.unwrap_and() {
            match c {
                Ambiguity(()) | PlaceholderTyOutlives(..) | AliasTyOutlivesViaEnv(..) => {
                    assert!(max_universe(infcx, c.clone()) < u);
                    pulled_constraints.push(c.clone());
                }
                RegionOutlives(region_1, region_2, ()) => {
                    let region_1_u = max_universe(infcx, region_1);
                    let region_2_u = max_universe(infcx, region_2);

                    if region_1_u != u && region_2_u != u {
                        pulled_constraints.push(c);
                        continue;
                    }

                    let assumptions = match assumptions {
                        Some(assumptions) => assumptions,
                        None => {
                            pulled_constraints.push(Ambiguity(()));
                            continue;
                        }
                    };

                    let mut candidates = vec![];

                    for ub in regions_outlived_by(region_1, assumptions)
                        .filter(|r| max_universe(infcx, *r) < u)
                    {
                        // FIXME(-Zassumptions-on-binders): if `region_2` is in a smaller universe there'll be both
                        // `'region_2` and `'static` as lower bounds which seems... unfortunate and may cause us to
                        // add a bunch of duplicate `'ub: 'static` candidates the more binders we leave.
                        for lb in regions_outliving(region_2, assumptions, infcx.cx())
                            .filter(|r| max_universe(infcx, *r) < u)
                        {
                            // As long as any region outlived by `region_1` outlives any region region which
                            // `region_2` outlives, we know that `region_1: region_2` holds. In other words,
                            // there exists some set of 4 regions for which `'r1: 'i1` `'i1: 'i2` `'i2: 'r2`
                            candidates.push(RegionOutlives(ub, lb, ()));
                        }
                    }

                    pulled_constraints.push(Or(candidates.into_boxed_slice()));
                }
                Or(_) | And(_) => unreachable!(),
            };
        }

        And(pulled_constraints.into_boxed_slice())
    };

    let ands = constraint.unwrap_or();
    Or(ands.into_iter().map(|and| pull_and(and)).collect::<Vec<_>>().into_boxed_slice())
}

/// Converts type outlives constraints into region outlives constraints. This assumes the *complete* set of
/// assumptions are known. This should not be called until the end of type checking.
///
/// The returned region constraint will not have *any* PlaceholderTyOutlives or AliasTyOutlivesViaEnv constraints.
pub fn destructure_type_outlives_constraints_in_root<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    S: Clone + std::fmt::Debug,
>(
    infcx: &Infcx,
    constraint: RegionConstraint<I, S>,
    assumptions: &Assumptions<I>,
) -> RegionConstraint<I, S> {
    use RegionConstraint::*;

    let destructure_and = |and: RegionConstraint<I, S>| {
        debug!("rewriting and {:?}", and);

        let mut destructured_constraints = Vec::new();
        for c in and.unwrap_and() {
            match c {
                Ambiguity(_) | RegionOutlives(..) => destructured_constraints.push(c),
                PlaceholderTyOutlives(ty, r, span) => destructured_constraints.push(Or(
                    regions_outlived_by_placeholder(ty, assumptions, infcx.cx())
                        .map(move |assumption_r| RegionOutlives(assumption_r, r, span.clone()))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )),
                AliasTyOutlivesViaEnv(bound_outlives, span) => destructured_constraints.push(
                    alias_outlives_candidates_from_assumptions(infcx, bound_outlives, assumptions)
                        .with_span(span),
                ),
                And(_) | Or(_) => unreachable!(),
            }
        }
        debug!(?destructured_constraints);
        And(destructured_constraints.into_boxed_slice())
    };

    let ands = constraint.unwrap_or();
    Or(ands.into_iter().map(|and| destructure_and(and)).collect::<Vec<_>>().into_boxed_slice())
}

/// Converts type outlives constraints into either region outlives constraints, or type outlives
/// constraints which do not contain anything from `u`.
///
/// This only works off assumptions associated with the binder corresponding to `u` both for
/// perf reasons and because the full set of region assumptions is not known during type checking
/// due to closure signature inference.
///
/// This only really causes problems for higher-ranked outlives assumptions, for example if we have
/// `where for<'a> <T as Trait<'a>>::Assoc: 'b` then we can't use that to prove `<T as Trait<'!c>>::Assoc: 'b`
/// until we are in the root context. See comments inside this function for more detail.
#[instrument(level = "debug", skip(infcx), ret)]
fn rewrite_type_outlives_constraints_in_universe_for_eager_placeholder_handling<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    constraint: RegionConstraint<I>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> RegionConstraint<I> {
    assert!(
        max_universe(infcx, constraint.clone()) <= u,
        "constraint {:?} contains terms from a larger universe than {:?}",
        constraint.clone(),
        u
    );

    use RegionConstraint::*;
    let rewrite_and = |and: RegionConstraint<I>| {
        let mut rewritten_constraints = Vec::new();
        for c in and.unwrap_and() {
            match c {
                Ambiguity(()) | RegionOutlives(..) => rewritten_constraints.push(c),
                PlaceholderTyOutlives(ty, region, ()) => {
                    rewritten_constraints.push(rewrite_placeholder_ty_outlives_constraints_in_universe_for_eager_placeholder_handling(
                        infcx,
                        ty,
                        region,
                        u,
                        assumptions,
                    ));
                }
                AliasTyOutlivesViaEnv(bound_outlives, ()) => {
                    rewritten_constraints.push(rewrite_alias_ty_outlives_constraints_in_universe_for_eager_placeholder_handling(
                        infcx,
                        bound_outlives,
                        u,
                        assumptions,
                    ))
                }
                And(_) | Or(_) => unreachable!(),
            }
        }

        And(rewritten_constraints.into_boxed_slice())
    };

    let ands = constraint.unwrap_or();
    Or(ands.into_iter().map(|and| rewrite_and(and)).collect::<Vec<_>>().into_boxed_slice())
}

fn rewrite_placeholder_ty_outlives_constraints_in_universe_for_eager_placeholder_handling<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    ty: I::Ty,
    region: Region<I>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> RegionConstraint<I> {
    use RegionConstraint::*;

    let ty_u = max_universe(infcx, ty);
    let region_u = max_universe(infcx, region);

    if region_u != u && ty_u != u {
        return PlaceholderTyOutlives(ty, region, ());
    }

    let assumptions = match assumptions {
        Some(assumptions) => assumptions,
        None => return Ambiguity(()),
    };

    let mut candidates = vec![];

    // There could be `!T: 'region` assumptions in the env even if `!T` is in a
    // smaller universe
    candidates.extend(
        regions_outlived_by_placeholder(ty, assumptions, infcx.cx())
            .map(move |assumption_r| RegionOutlives(assumption_r, region, ())),
    );

    // We can express `!T: 'region` as `!T: 'r` where `'r: 'region`. This is only necessary
    // if the placeholder type is in a smaller universe as otherwise we know all regions which
    // the placeholder outlives and can just destructure into an OR of RegionOutlives.
    if region_u == u && ty_u < u {
        candidates.extend(
            regions_outliving::<I>(region, assumptions, infcx.cx())
                .filter(|r| max_universe(infcx, *r) < u)
                .map(|r| PlaceholderTyOutlives(ty, r, ())),
        );
    }

    Or(candidates.into_boxed_slice())
}

fn rewrite_alias_ty_outlives_constraints_in_universe_for_eager_placeholder_handling<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    bound_outlives: Binder<I, (AliasTy<I>, Region<I>)>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> RegionConstraint<I> {
    use RegionConstraint::*;

    let mut candidates = Vec::new();

    // given there can be higher ranked assumptions, e.g. `for<'a> <T as Trait<'a>>::Assoc: 'c`, that
    // means that it's actually *always* possible for an alias outlive to be satisfied in the root universe
    // which means there should *always* be atleast two candidates when destructuring alias outlives. The
    // two candidates being component outlives and then a higher ranked alias outlives.
    //
    // we dont care about this for region outlives as `for<'a> 'a: 'b` can't exist as we don't elaborate
    // higher ranked type outlives assumptions into higher ranked region outlives assumptions. similarly,
    // we don't care about `for<'a> Foo<'a>: 'b` as we always destructure adts into their components and if
    // we dont equivalently elaborate the assumption into assumptions on the adt's components we just drop the
    // assumptions
    //
    // so actually only `for<'a, 'b> Alias<'a>: 'b` and `for<'a> T: 'a` are assumptions we actually need to
    // handle.
    //
    // we don't care about this when rewriting in the root universe as we know the complete set of assumptions
    if max_universe(infcx, bound_outlives) == u {
        let mut replacer = PlaceholderReplacer {
            cx: infcx.cx(),
            existing_var_count: bound_outlives.bound_vars().len(),
            bound_vars: IndexMap::default(),
            universe: u,
            current_index: DebruijnIndex::ZERO,
        };
        let escaping_outlives = bound_outlives.skip_binder().fold_with(&mut replacer);
        let bound_vars = bound_outlives.bound_vars().iter().chain(
            core::mem::take(&mut replacer.bound_vars)
                .into_iter()
                .map(|(_, bound_region)| BoundVariableKind::Region(bound_region.kind)),
        );
        let bound_outlives = Binder::bind_with_vars(
            escaping_outlives,
            I::BoundVarKinds::from_vars(infcx.cx(), bound_vars),
        );
        let candidate = RegionConstraint::AliasTyOutlivesViaEnv(bound_outlives, ());
        if max_universe(infcx, candidate.clone()) < u {
            candidates.push(candidate);
        } else {
            // `PlaceholderReplacer` only folds regions. A non-lifetime binder can leave
            // a placeholder type in `u`, so this type-outlives constraint cannot be
            // handled by the region-outlives-only eager placeholder machinery.
            candidates.push(Ambiguity(()));
        }
    }

    let assumptions = match assumptions {
        Some(assumptions) => assumptions,
        None => {
            candidates.push(Ambiguity(()));
            return Or(candidates.into_boxed_slice());
        }
    };

    // Actually look at the assumptions and matching our higher ranked alias outlives goal
    // against potentially higher ranked type outlives assumptions.
    candidates.push(alias_outlives_candidates_from_assumptions(infcx, bound_outlives, assumptions));

    // we can rewrite `Alias_u1: 'u2` into `Or(Alias_u1: 'u1)`
    // given a list of regions which outlive `'u2`
    //
    // we don't care about this when rewriting in the root universe as we know the complete set of assumptions
    let (escaping_alias, escaping_r) = bound_outlives.skip_binder();
    if max_universe(infcx, escaping_r) == u {
        let mut replacer = PlaceholderReplacer {
            cx: infcx.cx(),
            existing_var_count: bound_outlives.bound_vars().len(),
            bound_vars: IndexMap::default(),
            universe: u,
            current_index: DebruijnIndex::ZERO,
        };
        let escaping_alias = escaping_alias.fold_with(&mut replacer);
        let bound_vars = bound_outlives.bound_vars().iter().chain(
            core::mem::take(&mut replacer.bound_vars)
                .into_iter()
                .map(|(_, bound_region)| BoundVariableKind::Region(bound_region.kind)),
        );
        let bound_alias = Binder::bind_with_vars(
            escaping_alias,
            I::BoundVarKinds::from_vars(infcx.cx(), bound_vars),
        );

        // while we did skip the binder, bound vars aren't in any universe so
        // this can't be an escaping bound var
        for r2 in regions_outliving(escaping_r, assumptions, infcx.cx())
            .filter(|r2| max_universe(infcx, *r2) < u)
        {
            let candidate = AliasTyOutlivesViaEnv(bound_alias.map_bound(|alias| (alias, r2)), ());
            if max_universe(infcx, candidate.clone()) < u {
                candidates.push(candidate);
            } else {
                candidates.push(Ambiguity(()));
            }
        }
    }

    // I'm not convinced our handling here is *complete* so for now
    // let's be conservative and not let alias outlives' cause NoSolution
    // in coherence
    match infcx.typing_mode_raw() {
        TypingMode::Coherence => candidates.push(RegionConstraint::Ambiguity(())),
        TypingMode::Typeck { .. }
        | TypingMode::ErasedNotCoherence { .. }
        | TypingMode::PostTypeckUntilBorrowck { .. }
        | TypingMode::PostBorrowck { .. }
        | TypingMode::Reflection
        | TypingMode::PostAnalysis
        | TypingMode::Codegen => (),
    };

    RegionConstraint::Or(candidates.into_boxed_slice())
}

/// Returns all regions `r2` for which `r: r2` is known to hold in
/// the universe associated with `assumptions`
pub fn regions_outlived_by<I: Interner>(
    r: Region<I>,
    assumptions: &Assumptions<I>,
) -> impl Iterator<Item = Region<I>> {
    // FIXME(-Zassumptions-on-binders): do we need to be adding the reflexive edge here?
    assumptions.region_outlives.reachable_from(r).into_iter().chain([r])
}

/// Returns all regions `r2` for which `r2: r` is known to hold in
/// the universe associated with `assumptions`
pub fn regions_outliving<I: Interner>(
    r: Region<I>,
    assumptions: &Assumptions<I>,
    cx: I,
) -> impl Iterator<Item = Region<I>> {
    assumptions
        .inverse_region_outlives
        .reachable_from(r)
        .into_iter()
        // FIXME(-Zassumptions-on-binders): 'static may have been an input region canonicalized to something else is that important?
        // FIXME(-Zassumptions-on-binders): do we need to adding the reflexive edge here?
        .chain([r, Region::new_static(cx)])
}

/// Returns all regions `r` for which `!t: r` is known to hold in
/// the universe associated with `assumptions`
pub fn regions_outlived_by_placeholder<I: Interner>(
    t: I::Ty,
    assumptions: &Assumptions<I>,
    cx: I,
) -> impl Iterator<Item = Region<I>> {
    match t.kind() {
        TyKind::Placeholder(..) | TyKind::Param(..) => (),
        _ => unreachable!("non-placeholder in `regions_outlived_by_placeholder`: {t:?}"),
    }

    assumptions.type_outlives.iter().flat_map(move |binder| match binder.no_bound_vars() {
        Some(OutlivesClause(ty, r)) => (ty == t).then_some(r),
        None => Some(Region::new_static(cx)),
    })
}

pub struct PlaceholderReplacer<I: Interner> {
    cx: I,
    existing_var_count: usize,
    bound_vars: IndexMap<BoundVar, BoundRegion<I>>,
    universe: UniverseIndex,
    current_index: DebruijnIndex,
}

impl<I: Interner> TypeFolder<I> for PlaceholderReplacer<I> {
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_region(&mut self, r: Region<I>) -> Region<I> {
        match r.kind() {
            RegionKind::RePlaceholder(p) if p.universe == self.universe => {
                let bound_vars_len = self.bound_vars.len();
                let mapped_var = self.bound_vars.entry(p.bound.var).or_insert(BoundRegion {
                    var: BoundVar::from_usize(self.existing_var_count + bound_vars_len),
                    kind: p.bound.kind,
                });
                Region::new_bound(self.cx, self.current_index, *mapped_var)
            }
            // FIXME(-Zassumptions-on-binders): We should be handling region variables here somehow
            _ => r,
        }
    }

    fn fold_binder<T: TypeFoldable<I>>(&mut self, b: Binder<I, T>) -> Binder<I, T> {
        self.current_index.shift_in(1);
        let b = b.super_fold_with(self);
        self.current_index.shift_out(1);
        b
    }
}

/// Converts an `AliasTyOutlivesViaEnv` constraint into an OR of region outlives constraints by
/// matching the alias against any `Alias: 'a` assumptions. This is somewhat tricky as we have a
/// potentially higher ranked alias being equated with a potentially higher ranked assumption and
/// we don't handle it correctly right now (though it is a somewhat reasonable halfway step).
#[instrument(level = "debug", skip(infcx), ret)]
fn alias_outlives_candidates_from_assumptions<Infcx: InferCtxtLike<Interner = I>, I: Interner>(
    infcx: &Infcx,
    bound_outlives: Binder<I, (AliasTy<I>, Region<I>)>,
    assumptions: &Assumptions<I>,
) -> RegionConstraint<I> {
    let mut candidates = Vec::new();

    let prev_universe = infcx.universe();

    infcx.enter_forall_with_empty_assumptions(bound_outlives, |(alias, r)| {
        for bound_type_outlives in assumptions.type_outlives.iter() {
            let OutlivesClause(alias2, r2) =
                infcx.instantiate_binder_with_infer(*bound_type_outlives);

            let mut relation = HigherRankedAliasMatcher {
                infcx,
                region_constraints: vec![RegionConstraint::RegionOutlives(r2, r, ())],
            };

            // FIXME(#155345): Both sides should be rigid in the future.
            // Currently we can't guarantee that.
            if let Ok(_) = relation.relate(
                alias.to_ty(infcx.cx(), IsRigid::No),
                set_aliases_to_non_rigid(infcx.cx(), alias2).skip_norm_wip(),
            ) {
                candidates
                    .push(RegionConstraint::And(relation.region_constraints.into_boxed_slice()));
            }
        }
    });

    let constraint = RegionConstraint::Or(candidates.into_boxed_slice());

    let largest_universe = infcx.universe();
    debug!(?prev_universe, ?largest_universe);

    ((prev_universe.index() + 1)..=largest_universe.index())
        .map(|u| UniverseIndex::from_usize(u))
        .rev()
        .fold(constraint, |constraint, u| {
            eagerly_handle_placeholders_in_universe(infcx, constraint, u)
        })
}

struct HigherRankedAliasMatcher<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> {
    infcx: &'a Infcx,
    region_constraints: Vec<RegionConstraint<I>>,
}

impl<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> TypeRelation<I>
    for HigherRankedAliasMatcher<'a, Infcx, I>
{
    fn cx(&self) -> I {
        self.infcx.cx()
    }

    fn relate_ty_args(
        &mut self,
        a_ty: I::Ty,
        _b_ty: I::Ty,
        _ty_def_id: I::DefId,
        a_args: I::GenericArgs,
        b_args: I::GenericArgs,
        _mk: impl FnOnce(I::GenericArgs) -> I::Ty,
    ) -> RelateResult<I, I::Ty> {
        rustc_type_ir::relate::relate_args_invariantly(self, a_args, b_args)?;
        Ok(a_ty)
    }

    fn relate_with_variance<T: Relate<I>>(
        &mut self,
        _variance: Variance,
        _info: VarianceDiagInfo<I>,
        a: T,
        b: T,
    ) -> RelateResult<I, T> {
        // FIXME(-Zassumptions-on-binders): bivariance is important for opaque type args so
        // we should actually handle variance in some way here.
        self.relate(a, b)
    }

    fn tys(&mut self, a: I::Ty, b: I::Ty) -> RelateResult<I, I::Ty> {
        rustc_type_ir::relate::structurally_relate_tys(self, a, b)
    }

    fn regions(&mut self, a: Region<I>, b: Region<I>) -> RelateResult<I, Region<I>> {
        if a != b {
            self.region_constraints.push(RegionConstraint::RegionOutlives(a, b, ()));
            self.region_constraints.push(RegionConstraint::RegionOutlives(b, a, ()));
        }
        Ok(a)
    }

    fn consts(&mut self, a: I::Const, b: I::Const) -> RelateResult<I, I::Const> {
        rustc_type_ir::relate::structurally_relate_consts(self, a, b)
    }

    fn binders<T>(&mut self, a: Binder<I, T>, b: Binder<I, T>) -> RelateResult<I, Binder<I, T>>
    where
        T: Relate<I>,
    {
        self.infcx.enter_forall_with_empty_assumptions(a, |a| {
            let u = self.infcx.universe();
            self.infcx.insert_placeholder_assumptions(u, Some(Assumptions::empty()));
            let b = self.infcx.instantiate_binder_with_infer(b);
            self.relate(a, b)
        })?;

        self.infcx.enter_forall_with_empty_assumptions(b, |b| {
            let u = self.infcx.universe();
            self.infcx.insert_placeholder_assumptions(u, Some(Assumptions::empty()));
            let a = self.infcx.instantiate_binder_with_infer(a);
            self.relate(a, b)
        })?;

        Ok(a)
    }
}
