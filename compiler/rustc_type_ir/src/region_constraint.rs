//! The bulk of the logic for implementing `-Zassumptions-on-binders`

use derive_where::derive_where;
use indexmap::IndexSet;
#[cfg(feature = "nightly")]
use rustc_data_structures::transitive_relation::{TransitiveRelation, TransitiveRelationBuilder};
#[cfg(feature = "nightly")]
use rustc_macros::StableHash_NoContext;
use rustc_type_ir_macros::{GenericTypeVisitable, TypeFoldable_Generic, TypeVisitable_Generic};
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
use crate::visit::TypeSuperVisitable;
use crate::{
    AliasTy, Binder, BoundRegion, BoundVar, BoundVariableKind, ConstKind, DebruijnIndex,
    InferCtxtLike, InferTy, Interner, OutlivesPredicate, RegionKind, TyKind, TypeFoldable,
    TypeFolder, TypeVisitable, TypeVisitor, TypingMode, UniverseIndex, Variance,
};

#[derive_where(Clone, Debug; I: Interner)]
pub struct Assumptions<I: Interner> {
    pub type_outlives: Vec<Binder<I, OutlivesPredicate<I, I::Ty>>>,
    pub region_outlives: TransitiveRelation<I::Region>,
    pub inverse_region_outlives: TransitiveRelation<I::Region>,
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
        type_outlives: Vec<Binder<I, OutlivesPredicate<I, I::Ty>>>,
        region_outlives: TransitiveRelation<I::Region>,
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

    pub fn new_from_inverse_region_outlives(
        type_outlives: Vec<Binder<I, OutlivesPredicate<I, I::Ty>>>,
        inverse_region_outlives: TransitiveRelation<I::Region>,
    ) -> Self {
        Self {
            region_outlives: {
                let mut builder = TransitiveRelationBuilder::default();
                for (r1, r2) in inverse_region_outlives.base_edges() {
                    builder.add(r2, r1);
                }
                builder.freeze()
            },
            type_outlives,
            inverse_region_outlives,
        }
    }
}

#[derive_where(Clone, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum LeafRegionConstraint<I: Interner> {
    Ambiguity,
    RegionOutlives(I::Region, I::Region),
    /// Requirement that a (potentially higher ranked) alias outlives some (potentially higher ranked)
    /// region due to an assumption in the environment. This cannot be satisfied via component outlives
    /// or item bounds.
    ///
    /// We cannot eagerly look at assumptions as we are usually working with an incomplete set of assumptions
    /// and there may wind up being assumptions we can use to prove this when we're in a smaller universe.
    ///
    /// We eagerly destructure alias outlives requirements into region outlives requirements corresponding to
    /// component outlives & item bound outlives rules, leaving only param env candidates.
    AliasTyOutlivesViaEnv(Binder<I, (AliasTy<I>, I::Region)>),
    /// This is an `I::Ty` for two reasons:
    /// 1. We need the type visitable impl to be able to `visit_ty` on this so canonicalization
    ///    knows about the placeholder
    /// 2. When exiting the trait solver there may be placeholder outlives corresponding to params
    ///    from the root universe. These need to be changed from a `Placeholder` to the original
    ///    `Param`.
    ///
    /// We cannot eagerly look at assumptions as we are usually working with an incomplete set of assumptions
    /// and there may wind up being assumptions we can use to prove this when we're in a smaller universe.
    PlaceholderTyOutlives(I::Ty, I::Region),
}

#[derive_where(Clone, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct Or<I: Interner>(pub Box<[And<I>]>);
impl<I: Interner> Or<I> {
    pub fn new_true() -> Self {
        Self(Box::new([And::new([])]))
    }

    pub fn is_true(&self) -> bool {
        // OR([AND([])])
        if let [and] = &*self.0
            && and.0.len() == 0
        {
            true
        } else {
            false
        }
    }

    pub fn new_false() -> Self {
        Self(Box::new([]))
    }

    pub fn is_false(&self) -> bool {
        // OR([])
        self.0.len() == 0
    }

    pub fn new(i: impl IntoIterator<Item = And<I>>) -> Self {
        let ands = i.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let mut new_ands: Vec<And<I>> = Vec::new();

        for and in ands {
            if new_ands.iter().all(|c| !c.is_and_equivalent_to(&and)) {
                new_ands.push(and)
            }
        }

        Self(new_ands.into_boxed_slice())
    }

    // pub fn new_and(a: Or<I>, b: Or<I>) -> Self {
    //     return a;
        
    //     // I think this returns false if either a or b is false?
    //     let mut ands = Vec::new();
    //     for b_and in b.0 {
    //         ands.extend(
    //             a.0.clone()
    //                 .into_iter()
    //                 .map(|a_and| And::new(a_and.0.into_iter().chain(b_and.0.clone()))),
    //         );
    //     }

    //     Or::new(ands)
    // }

    pub fn new_or(a: Or<I>, b: Or<I>) -> Self {
        Or::new(a.0.into_iter().chain(b.0))
    }
}

pub mod dummy {
    use super::*;
    
    impl<I: Interner> Or<I> {
        pub fn new_and(a: Or<I>, b: Or<I>) -> Self {
            if a.is_true() {
                return b;
            } else if b.is_true() {
                return a;
            } else if a == b {
                return a;
            };

            debug!("new_and: \na={:?}\nb={:?}", a, b);
            
            // I think this returns false if either a or b is false?
            let mut ands = Vec::new();
            for b_and in b.0 {
                debug!(?ands);
                ands.extend(
                    a.0.clone()
                        .into_iter()
                        .map(|a_and| And::new(a_and.0.into_iter().chain(b_and.0.clone()))),
                );
            }

            Or::new(ands)
        }
    }
}

#[derive_where(Clone, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct And<I: Interner>(pub Box<[LeafRegionConstraint<I>]>);
impl<I: Interner> And<I> {
    pub fn new(i: impl IntoIterator<Item = LeafRegionConstraint<I>>) -> Self {
        Self(
            i.into_iter()
                .collect::<IndexSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }

    fn is_and_equivalent_to(&self, other: &And<I>) -> bool {
        let this = self.clone().0;
        let other = other.clone().0;

        this.iter().all(|c1| other.iter().any(|c2| c1 == c2))
            && other.iter().all(|c2| this.iter().any(|c1| c1 == c2))
    }
}

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
/// CanonicalFormRegionConstraints always have constraints shared between every OR element moved
/// into the and_constraint. Additionally they are always in "OR of AND of LEAF" form instead of
/// supporting arbitrary nesting of ORs/ANDs.
///
/// We also guarantee that there are no duplicate constraints in any of the `And` or `Or`s, though,
/// this is handled when constructing And/Ors rather than when constructing `CanonicalFormRegionConstraint`.
pub struct CanonicalFormRegionConstraint<I: Interner> {
    pub and_constraint: And<I>,
    pub or_constraint: Or<I>,
}

impl<I: Interner> CanonicalFormRegionConstraint<I> {
    pub fn new_from_or(or: Or<I>) -> Self {
        let Some(fst) = or.0.get(0).clone() else {
            return CanonicalFormRegionConstraint::new_false();
        };
        let mut and_constraint = fst.0.to_vec();

        for and in or.0.clone() {
            and_constraint.retain(|c| and.0.iter().any(|c2| c == c2));
        }
        let and_constraint = And::new(and_constraint);

        let or_constraint = Or::new(or.0.into_iter().map(|and| {
            And::new(and.0.into_iter().filter(|c| and_constraint.0.iter().all(|s_c| c != s_c)))
        }));

        Self { and_constraint, or_constraint }
    }

    pub fn splatted_and_constraints(&self) -> Or<I> {
        Or::new(self.or_constraint.0.iter().map(|and| {
            And::new(and.0.iter().cloned().chain(self.and_constraint.0.iter().cloned()))
        }))
    }

    pub fn new_and(
        a: CanonicalFormRegionConstraint<I>,
        b: CanonicalFormRegionConstraint<I>,
    ) -> Self {
        let and_constraint = And::new(a.and_constraint.0.into_iter().chain(b.and_constraint.0));
        let or_constraint = Or::new_and(a.or_constraint, b.or_constraint);

        Self { and_constraint, or_constraint }
    }

    pub fn new_true() -> Self {
        Self { and_constraint: And::new([]), or_constraint: Or::new_true() }
    }

    pub fn is_true(&self) -> bool {
        self.and_constraint.0.is_empty() && self.or_constraint.is_true()
    }

    pub fn new_false() -> Self {
        Self { and_constraint: And::new([]), or_constraint: Or::new_false() }
    }

    pub fn is_false(&self) -> bool {
        self.or_constraint.is_false()
    }

    pub fn new_ambig() -> Self {
        Self {
            and_constraint: And::new([LeafRegionConstraint::Ambiguity]),
            or_constraint: Or::new_true(),
        }
    }

    pub fn is_ambig(&self) -> bool {
        if let [c] = &*self.and_constraint.0
            && c.is_ambig()
            && self.or_constraint.is_true()
        {
            true
        } else {
            false
        }
    }
}

impl<I: Interner> Default for CanonicalFormRegionConstraint<I> {
    fn default() -> Self {
        Self::new_true()
    }
}

impl<I: Interner> LeafRegionConstraint<I> {
    pub fn is_ambig(&self) -> bool {
        matches!(self, Self::Ambiguity)
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
    constraint: CanonicalFormRegionConstraint<I>,
    u: UniverseIndex,
) -> CanonicalFormRegionConstraint<I> {
    let assumptions = infcx.get_placeholder_assumptions(u);

    // 1. rewrite type outlives constraints involving things from `u` into either region constraints
    //     involving things from `u` or type outlives constraints not involving things from `u`
    //
    //    IOW, we only want to encounter things from `u` as part of region out lives constraints.
    let constraint = rewrite_type_outlives_constraints_in_universe_for_eager_placeholder_handling(
        infcx,
        constraint,
        u,
        &assumptions,
    );

    // 2. compute transitive region outlives and get a new set of region outlives constraints by
    //     looking for every region which either a placeholder_u flows into it, or it flows into
    //     the placeholder.
    let constraint = compute_new_region_constraints(infcx, constraint, u);

    // 3. rewrite region outlives constraints (potentially to false/true)
    let constraint =
        pull_region_outlives_constraints_out_of_universe(infcx, constraint, u, &assumptions);

    // 4. actually evaluate the constraint to eagerly error on false
    evaluate_solver_constraint(constraint)
}

#[instrument(level = "debug", skip(infcx), ret)]
pub fn eagerly_handle_placeholders_in_root<Infcx: InferCtxtLike<Interner = I>, I: Interner>(
    infcx: &Infcx,
    constraint: CanonicalFormRegionConstraint<I>,
    assumptions: &Assumptions<I>,
) -> CanonicalFormRegionConstraint<I> {
    // 1. rewrite type outlives constraints into region constraints
    let constraint = rewrite_type_outlives_constraints_in_root(infcx, constraint, &assumptions);

    // 2. compute transitive region outlives and get a new set of region outlives constraints by
    //     looking for every region which either a placeholder_u flows into it, or it flows into
    //     the placeholder.
    let constraint = compute_new_region_constraints(infcx, constraint, UniverseIndex::ROOT);

    // 3. rewrite region outlives constraints (potentially to false/true)
    let constraint = pull_region_outlives_constraints_out_of_universe(
        infcx,
        constraint,
        UniverseIndex::ROOT,
        &Some(assumptions.clone()),
    );

    // 4. actually evaluate the constraint to eagerly error on false
    evaluate_solver_constraint(constraint)
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
    constraint: CanonicalFormRegionConstraint<I>,
    u: UniverseIndex,
) -> CanonicalFormRegionConstraint<I> {
    use LeafRegionConstraint::*;

    let extend_with_and = |builder: &mut TransitiveRelationBuilder<_>,
                           regions: &mut IndexSet<_>,
                           constraints: &mut Vec<_>,
                           and: &And<I>| {
        for c in &and.0 {
            match c {
                Ambiguity | PlaceholderTyOutlives(..) | AliasTyOutlivesViaEnv(..) => {
                    constraints.push(c.clone())
                }
                RegionOutlives(r1, r2) => {
                    regions.insert(*r1);
                    regions.insert(*r2);
                    builder.add(*r2, *r1);
                }
            }
        }
    };

    let mut base_region_flows_builder = TransitiveRelationBuilder::default();
    let mut base_regions = IndexSet::new();
    let mut base_constraints = Vec::new();
    extend_with_and(
        &mut base_region_flows_builder,
        &mut base_regions,
        &mut base_constraints,
        &constraint.and_constraint,
    );

    let mut new_ands = Vec::new();
    for and in &constraint.or_constraint.0 {
        let mut region_flows_builder = base_region_flows_builder.clone();
        let mut regions = base_regions.clone();
        let mut constraints = base_constraints.clone();
        extend_with_and(&mut region_flows_builder, &mut regions, &mut constraints, and);

        let region_flow = region_flows_builder.freeze();
        for r in regions.into_iter() {
            for ub in region_flow.reachable_from(r) {
                // we want to retain any region constraints between two "placeholder-likes" where for our
                // purposes a placeholder-like is either a placeholder or variable in a lower universe
                let is_placeholder_like = |r: I::Region| match r.kind() {
                    RegionKind::ReLateParam(..)
                    | RegionKind::ReEarlyParam(..)
                    | RegionKind::RePlaceholder(..)
                    | RegionKind::ReStatic => true,
                    RegionKind::ReVar(..) => max_universe(infcx, r) < u,
                    RegionKind::ReError(..) => false,
                    RegionKind::ReErased | RegionKind::ReBound(..) => unreachable!(),
                };

                if is_placeholder_like(r) && is_placeholder_like(ub) {
                    constraints.push(RegionOutlives(ub, r));
                }
            }
        }

        new_ands.push(Or::new([And::new(constraints)]))
    }

    CanonicalFormRegionConstraint::new_from_or(
        new_ands.into_iter().fold(Or::new_false(), |acc, c| Or::new_or(acc, c)),
    )
}

/// Evaluate ANDs and ORs to true/false/ambiguous based on whether their arguments are true/false/ambiguous
#[instrument(level = "debug", ret)]
pub fn evaluate_solver_constraint<I: Interner>(
    constraint: CanonicalFormRegionConstraint<I>,
) -> CanonicalFormRegionConstraint<I> {
    // TODO: this doesn't feel right... but also `CanonicalFormRegionConstraint` will:
    // - never have true/false in `and_constraint`
    // - never have "nested" falses in `or_constraint` so they must be just an empty `or_constraint`
    // - though it may have `true`s in the OR which can be folded away
    //
    // I guess this can't be moved into `CanonicalFormRegionConstraint` construction because we don't
    // want to eagerly turn the whole constraint ambiguous if we're going to add more constraints later
    // that may make it not-ambiguous? Actually that feels wrong it's probably fine... maybe this should
    // just be in `CanonicalFormRegionConstraint` lol.
    if constraint.and_constraint.0.iter().any(|c| c.is_ambig()) {
        return CanonicalFormRegionConstraint::new_ambig();
    }

    let mut found_ambig = false;
    for and in constraint.or_constraint.0.iter() {
        if and.0.is_empty() {
            return CanonicalFormRegionConstraint::new_from_or(Or::new([constraint.and_constraint]));
        }

        if and.0.iter().any(|c| c.is_ambig()) {
            found_ambig = true;
        }
    }

    if found_ambig {
        return CanonicalFormRegionConstraint::new_ambig();
    }

    constraint
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
    constraint: CanonicalFormRegionConstraint<I>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> CanonicalFormRegionConstraint<I> {
    assert!(max_universe(infcx, constraint.clone()) <= u);

    // FIXME(-Zassumptions-on-binders): we don't lower universes of region variables when exiting `u`
    // this seems dubious/potentially wrong? we can't just blindly do this though as if we had something
    // like `!T_u -> ?x_u -> !U_u` then lowering `?x` to `u-1` when exiting `u` would be wrong.
    //
    // I'm not even sure this would be necessary given we filter out region constraints involving regions#
    // from the current universe and only retain those between placeholders.

    use LeafRegionConstraint::*;

    let pull_and = |and: And<I>| {
        let mut ors = Vec::new();
        'outer: for c in and.0 {
            match c {
                Ambiguity | PlaceholderTyOutlives(..) | AliasTyOutlivesViaEnv(..) => {
                    assert!(max_universe(infcx, c.clone()) < u);
                    ors.push(Or::new([And::new([c.clone()])]));
                }
                RegionOutlives(region_1, region_2) => {
                    let region_1_u = max_universe(infcx, region_1);
                    let region_2_u = max_universe(infcx, region_2);

                    if region_1_u != u && region_2_u != u {
                        ors.push(Or::new([And::new([c])]));
                        continue;
                    }

                    let assumptions = match assumptions {
                        Some(assumptions) => assumptions,
                        None => {
                            ors.push(Or::new([And::new([Ambiguity])]));
                            continue;
                        }
                    };

                    let mut candidates = vec![];

                    // FIXME(-Zassumptions-on-binders): if `region_2` is in a smaller universe there'll be both
                    // `'region_2` and `'static` as lower bounds which seems... unfortunate and may cause us to
                    // add a bunch of duplicate `'ub: 'static` candidates the more binders we leave.
                    for ub in regions_outlived_by(region_1, assumptions) {
                        for lb in regions_outliving(region_2, assumptions, infcx.cx()) {
                            debug!("pair: {:?} {:?}", region_1, region_2);

                            // FIXME: `contains` not doing reflexive is fun
                            if assumptions.region_outlives.contains(ub, lb) || ub == lb {
                                continue 'outer;
                            }

                            if max_universe(infcx, ub) < u && max_universe(infcx, lb) < u {
                                // As long as any region outlived by `region_1` outlives any region region which
                                // `region_2` outlives, we know that `region_1: region_2` holds. In other words,
                                // there exists some set of 4 regions for which `'r1: 'i1` `'i1: 'i2` `'i2: 'r2`
                                candidates.push(RegionOutlives(ub, lb));
                            }
                        }
                    }

                    ors.push(Or::new(candidates.into_iter().map(|c| And::new([c]))));
                }
            };
        }

        ors.into_iter().fold(Or::new_true(), |acc, c| Or::new_and(acc, c))
    };

    let and_constraint = pull_and(constraint.and_constraint);
    let or_constraint = constraint
        .or_constraint
        .0
        .into_iter()
        .fold(Or::new_false(), |acc, c| Or::new_or(acc, pull_and(c)));
    CanonicalFormRegionConstraint::new_from_or(Or::new_and(and_constraint, or_constraint))
}

/// Converts type outlives constraints into region outlives constraints. This assumes the *complete* set of
/// assumptions are known. This should not be called until the end of type checking.
///
/// The returned region constraint will not have *any* PlaceholderTyOutlives or AliasTyOutlivesViaEnv constraints.
#[instrument(level = "debug", skip(infcx), ret)]
pub fn rewrite_type_outlives_constraints_in_root<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    constraint: CanonicalFormRegionConstraint<I>,
    assumptions: &Assumptions<I>,
) -> CanonicalFormRegionConstraint<I> {
    use LeafRegionConstraint::*;

    let rewrite_and = |and: &And<I>| {
        debug!("rewriting and: {:?}", and);
        let mut ors = Vec::new();
        for c in &and.0 {
            match c {
                Ambiguity | RegionOutlives(..) => ors.push(Or::new([And::new([c.clone()])])),
                PlaceholderTyOutlives(ty, r) => ors.push(Or::new(
                    regions_outlived_by_placeholder(*ty, assumptions, infcx.cx())
                        .map(move |assumption_r| And::new([RegionOutlives(assumption_r, *r)])),
                )),
                AliasTyOutlivesViaEnv(bound_outlives) => {
                    ors.push(alias_outlives_candidates_from_assumptions(
                        infcx,
                        *bound_outlives,
                        assumptions,
                    ));
                }
            }
        }
        debug!(?ors);
        let merged_ors = ors.into_iter().fold(Or::new_true(), |acc, c| Or::new_and(acc, c));
        debug!(?merged_ors);
        merged_ors
    };

    let and_constraint = rewrite_and(&constraint.and_constraint);
    let or_constraint = constraint
        .or_constraint
        .0
        .into_iter()
        .fold(Or::new_false(), |acc, c| Or::new_or(acc, rewrite_and(&c)));

    CanonicalFormRegionConstraint::new_from_or(Or::new_and(and_constraint, or_constraint))
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
    constraint: CanonicalFormRegionConstraint<I>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> CanonicalFormRegionConstraint<I> {
    use LeafRegionConstraint::*;

    assert!(
        max_universe(infcx, constraint.clone()) <= u,
        "constraint {:?} contains terms from a larger universe than {:?}",
        constraint.clone(),
        u
    );

    let rewrite_and = |and: And<I>| {
        let mut ors = Vec::new();
        for c in and.0 {
            match c {
                Ambiguity | RegionOutlives(..) => ors.push(Or::new([And::new([c])])),
                PlaceholderTyOutlives(ty, region) => {
                    ors.push(rewrite_placeholder_ty_outlives_constraints_in_universe_for_eager_placeholder_handling(infcx, ty, region, u, assumptions));
                }
                AliasTyOutlivesViaEnv(bound_outlives) => {
                    ors.push(rewrite_alias_ty_outlives_constraints_in_universe_for_eager_placeholder_handling(infcx, bound_outlives, u, assumptions));
                }
            }
        }
        ors.into_iter().fold(Or::new_true(), |acc, c| Or::new_and(acc, c))
    };

    let and_constraint = rewrite_and(constraint.and_constraint);
    let or_constraint = constraint
        .or_constraint
        .0
        .into_iter()
        .fold(Or::new_false(), |acc, c| Or::new_or(acc, rewrite_and(c)));

    CanonicalFormRegionConstraint::new_from_or(Or::new_and(and_constraint, or_constraint))
}

fn rewrite_placeholder_ty_outlives_constraints_in_universe_for_eager_placeholder_handling<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    ty: I::Ty,
    region: I::Region,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> Or<I> {
    use LeafRegionConstraint::*;

    let ty_u = max_universe(infcx, ty);
    let region_u = max_universe(infcx, region);

    if region_u != u && ty_u != u {
        return Or::new([And::new([PlaceholderTyOutlives(ty, region)])]);
    }

    let assumptions = match assumptions {
        Some(assumptions) => assumptions,
        None => return Or::new([And::new([Ambiguity])]),
    };

    let mut candidates = vec![];

    // There could be `!T: 'region` assumptions in the env even if `!T` is in a
    // smaller universe
    candidates.extend(
        regions_outlived_by_placeholder(ty, assumptions, infcx.cx())
            .map(move |assumption_r| RegionOutlives(assumption_r, region)),
    );

    // We can express `!T: 'region` as `!T: 'r` where `'r: 'region`. This is only necessary
    // if the placeholder type is in a smaller universe as otherwise we know all regions which
    // the placeholder outlives and can just destructure into an OR of RegionOutlives.
    if region_u == u && ty_u < u {
        candidates.extend(
            regions_outliving::<I>(region, assumptions, infcx.cx())
                .filter(|r| max_universe(infcx, *r) < u)
                .map(|r| PlaceholderTyOutlives(ty, r)),
        );
    }

    Or::new(candidates.into_iter().map(|c| And::new([c])))
}

fn rewrite_alias_ty_outlives_constraints_in_universe_for_eager_placeholder_handling<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    bound_outlives: Binder<I, (AliasTy<I>, I::Region)>,
    u: UniverseIndex,
    assumptions: &Option<Assumptions<I>>,
) -> Or<I> {
    use LeafRegionConstraint::*;

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
        candidates.push(Or::new([And::new([AliasTyOutlivesViaEnv(bound_outlives)])]));
    }

    let assumptions = match assumptions {
        Some(assumptions) => assumptions,
        None => {
            candidates.push(Or::new([And::new([Ambiguity])]));
            return candidates.into_iter().fold(Or::new_false(), |acc, c| Or::new_or(acc, c));
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
        candidates.push(Or::new(
            regions_outliving(escaping_r, assumptions, infcx.cx())
                .filter(|r2| max_universe(infcx, *r2) < u)
                .map(|r2| {
                    And::new([AliasTyOutlivesViaEnv(bound_alias.map_bound(|alias| (alias, r2)))])
                }),
        ));
    }

    // I'm not convinced our handling here is *complete* so for now
    // let's be conservative and not let alias outlives' cause NoSolution
    // in coherence
    match infcx.typing_mode_raw() {
        TypingMode::Coherence => candidates.push(Or::new([And::new([Ambiguity])])),
        TypingMode::Typeck { .. }
        | TypingMode::ErasedNotCoherence { .. }
        | TypingMode::PostTypeckUntilBorrowck { .. }
        | TypingMode::PostBorrowck { .. }
        | TypingMode::PostAnalysis
        | TypingMode::Codegen => (),
    };

    candidates.into_iter().fold(Or::new_false(), |acc, c| Or::new_or(acc, c))
}

/// Returns all regions `r2` for which `r: r2` is known to hold in
/// the universe associated with `assumptions`
pub fn regions_outlived_by<I: Interner>(
    r: I::Region,
    assumptions: &Assumptions<I>,
) -> impl Iterator<Item = I::Region> {
    // FIXME(-Zassumptions-on-binders): do we need to be adding the reflexive edge here?
    assumptions.region_outlives.reachable_from(r).into_iter().chain([r])
}

/// Returns all regions `r2` for which `r2: r` is known to hold in
/// the universe associated with `assumptions`
pub fn regions_outliving<I: Interner>(
    r: I::Region,
    assumptions: &Assumptions<I>,
    cx: I,
) -> impl Iterator<Item = I::Region> {
    assumptions
        .inverse_region_outlives
        .reachable_from(r)
        .into_iter()
        // FIXME(-Zassumptions-on-binders): 'static may have been an input region canonicalized to something else is that important?
        // FIXME(-Zassumptions-on-binders): do we need to adding the reflexive edge here?
        .chain([r, I::Region::new_static(cx)])
}

/// Returns all regions `r` for which `!t: r` is known to hold in
/// the universe associated with `assumptions`
pub fn regions_outlived_by_placeholder<I: Interner>(
    t: I::Ty,
    assumptions: &Assumptions<I>,
    cx: I,
) -> impl Iterator<Item = I::Region> {
    match t.kind() {
        TyKind::Placeholder(..) | TyKind::Param(..) => (),
        _ => unreachable!("non-placeholder in `regions_outlived_by_placeholder`: {t:?}"),
    }

    assumptions.type_outlives.iter().flat_map(move |binder| match binder.no_bound_vars() {
        Some(OutlivesPredicate(ty, r)) => (ty == t).then_some(r),
        None => Some(I::Region::new_static(cx)),
    })
}

/// The largest universe a variable or placeholder was from in `t`
pub fn max_universe<Infcx: InferCtxtLike<Interner = I>, I: Interner, T: TypeVisitable<I>>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    let mut visitor = MaxUniverse::new(infcx);
    t.visit_with(&mut visitor);
    visitor.max_universe()
}

// FIXME(-Zassumptions-on-binders): Share this with the visitor used by generalization. We currently don't
// as generalization does not look at universes of inference variables but we do
struct MaxUniverse<'a, Infcx: InferCtxtLike> {
    max_universe: UniverseIndex,
    infcx: &'a Infcx,
}

impl<'a, Infcx: InferCtxtLike> MaxUniverse<'a, Infcx> {
    fn new(infcx: &'a Infcx) -> Self {
        MaxUniverse { infcx, max_universe: UniverseIndex::ROOT }
    }

    fn max_universe(self) -> UniverseIndex {
        self.max_universe
    }
}

impl<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> TypeVisitor<I>
    for MaxUniverse<'a, Infcx>
{
    type Result = ();

    fn visit_ty(&mut self, t: I::Ty) {
        match t.kind() {
            TyKind::Placeholder(p) => self.max_universe = self.max_universe.max(p.universe),
            TyKind::Infer(InferTy::TyVar(inf)) => {
                let u = self.infcx.universe_of_ty(inf).unwrap();
                debug!("var {inf:?} in universe {u:?}");
                self.max_universe = self.max_universe.max(u);
            }
            _ => t.super_visit_with(self),
        }
    }

    fn visit_const(&mut self, c: I::Const) {
        match c.kind() {
            ConstKind::Placeholder(p) => self.max_universe = self.max_universe.max(p.universe),
            ConstKind::Infer(rustc_type_ir::InferConst::Var(inf)) => {
                let u = self.infcx.universe_of_ct(inf).unwrap();
                debug!("var {inf:?} in universe {u:?}");
                self.max_universe = self.max_universe.max(u);
            }
            _ => c.super_visit_with(self),
        }
    }

    fn visit_region(&mut self, r: I::Region) {
        match r.kind() {
            RegionKind::RePlaceholder(p) => self.max_universe = self.max_universe.max(p.universe),
            RegionKind::ReVar(var) => {
                let u = self.infcx.universe_of_lt(var).unwrap();
                debug!("var {var:?} in universe {u:?}");
                self.max_universe = self.max_universe.max(u);
            }
            _ => (),
        }
    }
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

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        match r.kind() {
            RegionKind::RePlaceholder(p) if p.universe == self.universe => {
                let bound_vars_len = self.bound_vars.len();
                let mapped_var = self.bound_vars.entry(p.bound.var).or_insert(BoundRegion {
                    var: BoundVar::from_usize(self.existing_var_count + bound_vars_len),
                    kind: p.bound.kind,
                });
                I::Region::new_bound(self.cx, self.current_index, *mapped_var)
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
    bound_outlives: Binder<I, (AliasTy<I>, I::Region)>,
    assumptions: &Assumptions<I>,
) -> Or<I> {
    let mut candidates = Vec::new();

    let prev_universe = infcx.universe();

    infcx.enter_forall_with_empty_assumptions(bound_outlives, |(alias, r)| {
        for bound_type_outlives in assumptions.type_outlives.iter() {
            let OutlivesPredicate(alias2, r2) =
                infcx.instantiate_binder_with_infer(*bound_type_outlives);

            let mut relation = HigherRankedAliasMatcher {
                infcx,
                region_constraints: vec![LeafRegionConstraint::RegionOutlives(r2, r)],
            };

            if let Ok(_) = relation.relate(alias.to_ty(infcx.cx()), alias2) {
                candidates.push(And::new(relation.region_constraints));
            }
        }
    });

    let constraint = CanonicalFormRegionConstraint::new_from_or(Or::new(candidates));

    let largest_universe = infcx.universe();
    debug!(?prev_universe, ?largest_universe);

    let canonical_constraint = ((prev_universe.index() + 1)..=largest_universe.index())
        .map(|u| UniverseIndex::from_usize(u))
        .rev()
        .fold(constraint, |constraint, u| {
            eagerly_handle_placeholders_in_universe(infcx, constraint, u)
        });

    canonical_constraint.splatted_and_constraints()
}

struct HigherRankedAliasMatcher<'a, Infcx: InferCtxtLike<Interner = I>, I: Interner> {
    infcx: &'a Infcx,
    region_constraints: Vec<LeafRegionConstraint<I>>,
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

    fn regions(&mut self, a: I::Region, b: I::Region) -> RelateResult<I, I::Region> {
        if a != b {
            self.region_constraints.push(LeafRegionConstraint::RegionOutlives(a, b));
            self.region_constraints.push(LeafRegionConstraint::RegionOutlives(b, a));
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
