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
use crate::visit::TypeSuperVisitable;
use crate::{
    AliasTy, Binder, BoundRegion, BoundVar, BoundVariableKind, ConstKind, DebruijnIndex,
    FallibleTypeFolder, InferCtxtLike, InferTy, Interner, OutlivesPredicate, RegionKind, TyKind,
    TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor, TypingMode, UniverseIndex, Variance,
    VisitorResult,
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
}

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
pub enum RegionConstraint<I: Interner> {
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

    And(Box<[RegionConstraint<I>]>),
    Or(Box<[RegionConstraint<I>]>),
}

// This is not a derived impl because a perfect derive leads to inductive
// cycle causing the trait to never actually be implemented
#[cfg(feature = "nightly")]
impl<I: Interner> StableHash for RegionConstraint<I>
where
    I::Region: StableHash,
    I::Ty: StableHash,
    I::GenericArgs: StableHash,
    I::TraitAssocTyId: StableHash,
    I::InherentAssocTyId: StableHash,
    I::OpaqueTyId: StableHash,
    I::FreeTyAliasId: StableHash,
    I::BoundVarKinds: StableHash,
{
    #[inline]
    fn stable_hash<CTX: StableHashCtxt>(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        use RegionConstraint::*;

        std::mem::discriminant(self).stable_hash(hcx, hasher);
        match self {
            Ambiguity => (),
            RegionOutlives(a, b) => {
                a.stable_hash(hcx, hasher);
                b.stable_hash(hcx, hasher);
            }
            AliasTyOutlivesViaEnv(outlives) => {
                outlives.stable_hash(hcx, hasher);
            }
            PlaceholderTyOutlives(a, b) => {
                a.stable_hash(hcx, hasher);
                b.stable_hash(hcx, hasher);
            }
            And(and) => {
                for a in and.iter() {
                    a.stable_hash(hcx, hasher);
                }
            }
            Or(or) => {
                for a in or.iter() {
                    a.stable_hash(hcx, hasher);
                }
            }
        }
    }
}

impl<I: Interner> TypeFoldable<I> for RegionConstraint<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, f: &mut F) -> Result<Self, F::Error> {
        use RegionConstraint::*;
        Ok(match self {
            Ambiguity => self,
            RegionOutlives(a, b) => RegionOutlives(a.try_fold_with(f)?, b.try_fold_with(f)?),
            AliasTyOutlivesViaEnv(outlives) => AliasTyOutlivesViaEnv(outlives.try_fold_with(f)?),
            PlaceholderTyOutlives(a, b) => {
                PlaceholderTyOutlives(a.try_fold_with(f)?, b.try_fold_with(f)?)
            }
            And(and) => {
                let mut new_and = Vec::new();
                for a in and {
                    new_and.push(a.try_fold_with(f)?);
                }
                And(new_and.into_boxed_slice())
            }
            Or(or) => {
                let mut new_or = Vec::new();
                for a in or {
                    new_or.push(a.try_fold_with(f)?);
                }
                Or(new_or.into_boxed_slice())
            }
        })
    }

    fn fold_with<F: TypeFolder<I>>(self, f: &mut F) -> Self {
        use RegionConstraint::*;
        match self {
            Ambiguity => self,
            RegionOutlives(a, b) => RegionOutlives(a.fold_with(f), b.fold_with(f)),
            AliasTyOutlivesViaEnv(outlives) => AliasTyOutlivesViaEnv(outlives.fold_with(f)),
            PlaceholderTyOutlives(a, b) => PlaceholderTyOutlives(a.fold_with(f), b.fold_with(f)),
            And(and) => {
                let mut new_and = Vec::new();
                for a in and {
                    new_and.push(a.fold_with(f));
                }
                And(new_and.into_boxed_slice())
            }
            Or(or) => {
                let mut new_or = Vec::new();
                for a in or {
                    new_or.push(a.fold_with(f));
                }
                Or(new_or.into_boxed_slice())
            }
        }
    }
}

impl<I: Interner> TypeVisitable<I> for RegionConstraint<I> {
    fn visit_with<F: TypeVisitor<I>>(&self, f: &mut F) -> F::Result {
        use core::ops::ControlFlow::*;

        use RegionConstraint::*;

        match self {
            Ambiguity => (),
            RegionOutlives(a, b) => {
                if let b @ Break(_) = a.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
                if let b @ Break(_) = b.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
            }
            AliasTyOutlivesViaEnv(outlives) => {
                return outlives.visit_with(f);
            }
            PlaceholderTyOutlives(a, b) => {
                if let b @ Break(_) = a.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
                if let b @ Break(_) = b.visit_with(f).branch() {
                    return F::Result::from_branch(b);
                };
            }
            And(and) => {
                for a in and {
                    if let b @ Break(_) = a.visit_with(f).branch() {
                        return F::Result::from_branch(b);
                    };
                }
            }
            Or(or) => {
                for a in or {
                    if let b @ Break(_) = a.visit_with(f).branch() {
                        return F::Result::from_branch(b);
                    };
                }
            }
        };

        F::Result::output()
    }
}

impl<I: Interner> Default for RegionConstraint<I> {
    fn default() -> Self {
        Self::new_true()
    }
}

impl<I: Interner> RegionConstraint<I> {
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

    pub fn unwrap_or(self) -> Box<[RegionConstraint<I>]> {
        match self {
            Self::Or(ors) => ors,
            _ => panic!("`unwrap_or` on non-Or: {self:?}"),
        }
    }

    pub fn unwrap_and(self) -> Box<[RegionConstraint<I>]> {
        match self {
            Self::And(ands) => ands,
            _ => panic!("`unwrap_and` on non-And: {self:?}"),
        }
    }

    pub fn is_and(&self) -> bool {
        matches!(self, Self::And(_))
    }

    pub fn is_ambig(&self) -> bool {
        matches!(self, Self::Ambiguity)
    }

    pub fn and(self, other: RegionConstraint<I>) -> RegionConstraint<I> {
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

    /// Converts the region constraint into an ORs of ANDs of "leaf" constraints. Where
    /// a leaf constraint is a non-or/and constraint.
    #[instrument(level = "debug", ret)]
    pub fn canonical_form(self) -> Self {
        use RegionConstraint::*;

        fn permutations<I: Interner>(
            ors: &[Vec<RegionConstraint<I>>],
        ) -> Vec<Vec<RegionConstraint<I>>> {
            match ors {
                [] => vec![vec![]],
                [or1] => {
                    let mut choices = vec![];
                    for choice in or1 {
                        choices.push(vec![choice.clone()]);
                    }
                    choices
                }
                [or1, rest_ors @ ..] => {
                    let mut choices = vec![];
                    for choice in or1 {
                        choices.extend(permutations(rest_ors).into_iter().map(|mut and| {
                            and.push(choice.clone());
                            and
                        }));
                    }
                    choices
                }
            }
        }

        let canonical = match self {
            And(ands) => {
                // AND of OR of AND of LEAFs
                //
                // We can turn `AND of OR of X` into `OR of AND of X` by enumerating every set of choices
                // for the list of ORs. For example if we have `AND ( OR(A, B), OR(C, D) )` we can convert this into
                // `OR ( AND (A, C), AND (A, D), AND (B, C), AND (B, D ))`
                //
                // if A/B/C/D are all in canonical forms then we wind up with an `OR of AND of AND of LEAFs` which
                // is trivially canonicalizeable by flattening the multiple layers of AND into one.
                let ors = ands
                    .into_iter()
                    .map(|c| c.canonical_form().unwrap_or().to_vec())
                    .collect::<Vec<_>>();
                debug!(?ors);
                let or_permutations = permutations(&ors);
                debug!(?or_permutations);

                Or(or_permutations
                    .into_iter()
                    .map(|c| {
                        And(c
                            .into_iter()
                            .flat_map(|c2| c2.unwrap_and().into_iter())
                            .collect::<Vec<_>>()
                            .into_boxed_slice())
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice())
            }
            Or(ors) => {
                // OR of OR of AND of LEAFs
                //
                // trivially canonicalizeable by concatenating all of the ORs into one big OR
                Or(ors
                    .into_iter()
                    .flat_map(|c| c.canonical_form().unwrap_or().into_iter())
                    .collect::<Vec<_>>()
                    .into_boxed_slice())
            }
            _ => Or(Box::new([And(Box::new([self]))])),
        };

        assert!(
            canonical.is_canonical_form(),
            "non canonical form region constraint: {:?}",
            canonical
        );
        canonical
    }

    fn is_leaf_constraint(&self) -> bool {
        use RegionConstraint::*;
        match self {
            Ambiguity
            | RegionOutlives(..)
            | AliasTyOutlivesViaEnv(..)
            | PlaceholderTyOutlives(..) => true,
            And(..) | Or(..) => false,
        }
    }

    fn is_canonical_and(&self) -> bool {
        if let Self::And(ands) = self { ands.iter().all(|c| c.is_leaf_constraint()) } else { false }
    }

    pub fn is_canonical_form(&self) -> bool {
        if let Self::Or(ors) = self { ors.iter().all(|c| c.is_canonical_and()) } else { false }
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
    use RegionConstraint::*;

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

    // 2. rewrite the constraint into a canonical ORs of ANDs form
    let constraint = constraint.canonical_form();

    // 3. compute transitive region outlives and get a new set of region outlives constraints by
    //     looking for every region which either a placeholder_u flows into it, or it flows into
    //     the placeholder.
    //
    //    do this for each element in the top level OR
    let constraint = Or(constraint
        .unwrap_or()
        .into_iter()
        .map(|c| {
            let and =
                And(compute_new_region_constraints(infcx, &c.unwrap_and(), u).into_boxed_slice());

            // 4. rewrite region outlives constraints (potentially to false/true)
            pull_region_outlives_constraints_out_of_universe(infcx, and, u, &assumptions)
        })
        .collect::<Vec<_>>()
        .into_boxed_slice());

    // 5. actually evaluate the constraint to eagerly error on false
    evaluate_solver_constraint(&constraint)
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
    constraints: &[RegionConstraint<I>],
    u: UniverseIndex,
) -> Vec<RegionConstraint<I>> {
    use RegionConstraint::*;

    let mut new_constraints = vec![];

    let mut region_flows_builder = TransitiveRelationBuilder::default();
    let mut regions = IndexSet::new();
    for c in constraints {
        match c {
            And(..) | Or(..) => unreachable!(),
            Ambiguity | PlaceholderTyOutlives(..) | AliasTyOutlivesViaEnv(..) => {
                new_constraints.push(c.clone())
            }
            RegionOutlives(r1, r2) => {
                regions.insert(r1);
                regions.insert(r2);
                region_flows_builder.add(r2, r1);
            }
        }
    }

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

            if is_placeholder_like(*r) && is_placeholder_like(*ub) {
                new_constraints.push(RegionOutlives(*ub, *r));
            }
        }
    }

    new_constraints
}

/// Evaluate ANDs and ORs to true/false/ambiguous based on whether their arguments are true/false/ambiguous
#[instrument(level = "debug", ret)]
pub fn evaluate_solver_constraint<I: Interner>(
    constraint: &RegionConstraint<I>,
) -> RegionConstraint<I> {
    use RegionConstraint::*;
    match constraint {
        Ambiguity | RegionOutlives(..) | AliasTyOutlivesViaEnv(..) | PlaceholderTyOutlives(..) => {
            constraint.clone()
        }
        And(and) => {
            let mut and_constraints = Vec::new();
            let mut is_ambiguous_constraint = false;
            for c in and.iter() {
                let evaluated_constraint = evaluate_solver_constraint(c);
                if evaluated_constraint.is_true() {
                    // - do nothing
                } else if evaluated_constraint.is_false() {
                    return RegionConstraint::new_false();
                } else if evaluated_constraint.is_ambig() {
                    is_ambiguous_constraint = true;
                } else {
                    and_constraints.push(evaluated_constraint);
                }
            }

            if is_ambiguous_constraint {
                RegionConstraint::Ambiguity
            } else {
                RegionConstraint::And(and_constraints.into_boxed_slice())
            }
        }
        Or(or) => {
            let mut or_constraints = Vec::new();
            let mut is_ambiguous_constraint = false;
            for c in or.iter() {
                let evaluated_constraint = evaluate_solver_constraint(c);
                if evaluated_constraint.is_false() {
                    // do nothing
                } else if evaluated_constraint.is_true() {
                    return RegionConstraint::new_true();
                } else if evaluated_constraint.is_ambig() {
                    is_ambiguous_constraint = true;
                } else {
                    or_constraints.push(evaluated_constraint);
                }
            }

            if is_ambiguous_constraint {
                RegionConstraint::Ambiguity
            } else {
                RegionConstraint::Or(or_constraints.into_boxed_slice())
            }
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
    match constraint {
        Ambiguity | PlaceholderTyOutlives(..) | AliasTyOutlivesViaEnv(..) => {
            assert!(max_universe(infcx, constraint.clone()) < u);
            constraint
        }
        RegionOutlives(region_1, region_2) => {
            let region_1_u = max_universe(infcx, region_1);
            let region_2_u = max_universe(infcx, region_2);

            if region_1_u != u && region_2_u != u {
                return constraint;
            }

            let assumptions = match assumptions {
                Some(assumptions) => assumptions,
                None => return RegionConstraint::Ambiguity,
            };

            let mut candidates = vec![];
            for ub in
                regions_outlived_by(region_1, assumptions).filter(|r| max_universe(infcx, *r) < u)
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
                    candidates.push(RegionOutlives(ub, lb));
                }
            }

            RegionConstraint::Or(candidates.into_boxed_slice())
        }
        And(constraints) => And(constraints
            .into_iter()
            .map(|constraint| {
                pull_region_outlives_constraints_out_of_universe(infcx, constraint, u, assumptions)
            })
            .collect()),
        Or(_) => unreachable!(),
    }
}

/// Converts type outlives constraints into region outlives constraints. This assumes the *complete* set of
/// assumptions are known. This should not be called until the end of type checking.
///
/// The returned region constraint will not have *any* PlaceholderTyOutlives or AliasTyOutlivesViaEnv constraints.
pub fn destructure_type_outlives_constraints_in_root<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
>(
    infcx: &Infcx,
    constraint: RegionConstraint<I>,
    assumptions: &Assumptions<I>,
) -> RegionConstraint<I> {
    use RegionConstraint::*;

    match constraint {
        Ambiguity | RegionOutlives(..) => constraint,
        PlaceholderTyOutlives(ty, r) => {
            Or(regions_outlived_by_placeholder(ty, assumptions, infcx.cx())
                .map(move |assumption_r| RegionOutlives(assumption_r, r))
                .collect::<Vec<_>>()
                .into_boxed_slice())
        }
        AliasTyOutlivesViaEnv(bound_outlives) => {
            alias_outlives_candidates_from_assumptions(infcx, bound_outlives, assumptions)
        }
        And(constraints) => And(constraints
            .into_iter()
            .map(|constraint| {
                destructure_type_outlives_constraints_in_root(infcx, constraint, assumptions)
            })
            .collect()),
        Or(constraints) => Or(constraints
            .into_iter()
            .map(|constraint| {
                destructure_type_outlives_constraints_in_root(infcx, constraint, assumptions)
            })
            .collect()),
    }
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
    match constraint {
        Ambiguity | RegionOutlives(..) => constraint,
        PlaceholderTyOutlives(ty, region) => {
            let ty_u = max_universe(infcx, ty);
            let region_u = max_universe(infcx, region);

            if region_u != u && ty_u != u {
                return constraint;
            }

            let assumptions = match assumptions {
                Some(assumptions) => assumptions,
                None => return Ambiguity,
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

            Or(candidates.into_boxed_slice())
        }
        AliasTyOutlivesViaEnv(bound_outlives) => {
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
                candidates.push(RegionConstraint::AliasTyOutlivesViaEnv(bound_outlives));
            }

            let assumptions = match assumptions {
                Some(assumptions) => assumptions,
                None => {
                    candidates.push(Ambiguity);
                    return Or(candidates.into_boxed_slice());
                }
            };

            // Actually look at the assumptions and matching our higher ranked alias outlives goal
            // against potentially higher ranked type outlives assumptions.
            candidates.push(alias_outlives_candidates_from_assumptions(
                infcx,
                bound_outlives,
                assumptions,
            ));

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
                candidates.extend(
                    regions_outliving(escaping_r, assumptions, infcx.cx())
                        .filter(|r2| max_universe(infcx, *r2) < u)
                        .map(|r2| AliasTyOutlivesViaEnv(bound_alias.map_bound(|alias| (alias, r2))))
                        .collect::<Vec<_>>(),
                );
            }

            // I'm not convinced our handling here is *complete* so for now
            // let's be conservative and not let alias outlives' cause NoSolution
            // in coherence
            match infcx.typing_mode_raw() {
                TypingMode::Coherence => candidates.push(RegionConstraint::Ambiguity),
                TypingMode::Analysis { .. }
                | TypingMode::ErasedNotCoherence { .. }
                | TypingMode::Borrowck { .. }
                | TypingMode::PostBorrowckAnalysis { .. }
                | TypingMode::PostAnalysis => (),
            };

            RegionConstraint::Or(candidates.into_boxed_slice())
        }
        And(constraints) => And(constraints
            .into_iter()
            .map(|constraint| {
                rewrite_type_outlives_constraints_in_universe_for_eager_placeholder_handling(
                    infcx,
                    constraint,
                    u,
                    assumptions,
                )
            })
            .collect()),
        Or(constraints) => Or(constraints
            .into_iter()
            .map(|constraint| {
                rewrite_type_outlives_constraints_in_universe_for_eager_placeholder_handling(
                    infcx,
                    constraint,
                    u,
                    assumptions,
                )
            })
            .collect()),
    }
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
) -> RegionConstraint<I> {
    let mut candidates = Vec::new();

    let prev_universe = infcx.universe();

    // FIXME(-Zassumptions-on-binders): Handle the assumptions on this binder
    infcx.enter_forall(bound_outlives, |(alias, r)| {
        let u = infcx.universe();
        infcx.insert_placeholder_assumptions(u, Some(Assumptions::empty()));

        for bound_type_outlives in assumptions.type_outlives.iter() {
            let OutlivesPredicate(alias2, r2) =
                infcx.instantiate_binder_with_infer(*bound_type_outlives);

            let mut relation = HigherRankedAliasMatcher {
                infcx,
                region_constraints: vec![RegionConstraint::RegionOutlives(r2, r)],
            };

            if let Ok(_) = relation.relate(alias.to_ty(infcx.cx()), alias2) {
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

    fn regions(&mut self, a: I::Region, b: I::Region) -> RelateResult<I, I::Region> {
        if a != b {
            self.region_constraints.push(RegionConstraint::RegionOutlives(a, b));
            self.region_constraints.push(RegionConstraint::RegionOutlives(b, a));
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
        self.infcx.enter_forall(a, |a| {
            let u = self.infcx.universe();
            self.infcx.insert_placeholder_assumptions(u, Some(Assumptions::empty()));
            let b = self.infcx.instantiate_binder_with_infer(b);
            self.relate(a, b)
        })?;

        self.infcx.enter_forall(b, |b| {
            let u = self.infcx.universe();
            self.infcx.insert_placeholder_assumptions(u, Some(Assumptions::empty()));
            let a = self.infcx.instantiate_binder_with_infer(a);
            self.relate(a, b)
        })?;

        Ok(a)
    }
}
