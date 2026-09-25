use std::fmt;

pub(crate) use region_context::*;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::scc::Sccs;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_index::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_infer::infer::region_constraints::{GenericKind, VerifyBound};
use rustc_middle::mir::{ConstraintCategory, Local, Location};
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{self, RegionVid, TyCtxt};
use rustc_span::Span;
use tracing::{Level, debug, enabled, instrument};

use crate::BorrowckInferCtxt;
use crate::constraints::{ConstraintSccIndex, OutlivesConstraint, OutlivesConstraintSet};
use crate::handle_placeholders::RegionTracker;
use crate::type_check::Locations;

mod dump_mir;
mod graphviz;
pub(crate) mod opaque_types;
mod region_context;
mod reverse_sccs;

pub(crate) mod values;

/// The representative region variable for an SCC, tagged by its origin.
/// We prefer placeholders over existentially quantified variables, otherwise
/// it's the one with the smallest Region Variable ID. In other words,
/// the order of this enumeration really matters!
#[derive(Copy, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub(crate) enum Representative {
    FreeRegion(RegionVid),
    Placeholder(RegionVid),
    Existential(RegionVid),
}

impl Representative {
    pub(crate) fn rvid(self) -> RegionVid {
        match self {
            Representative::FreeRegion(region_vid)
            | Representative::Placeholder(region_vid)
            | Representative::Existential(region_vid) => region_vid,
        }
    }

    pub(crate) fn new(r: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        match definition.origin {
            NllRegionVariableOrigin::FreeRegion => Representative::FreeRegion(r),
            NllRegionVariableOrigin::Placeholder(_) => Representative::Placeholder(r),
            NllRegionVariableOrigin::Existential { .. } => Representative::Existential(r),
        }
    }
}

pub(crate) type ConstraintSccs = Sccs<RegionVid, ConstraintSccIndex>;

#[derive(Debug)]
pub(crate) struct RegionDefinition<'tcx> {
    /// What kind of variable is this -- a free region? existential
    /// variable? etc. (See the `NllRegionVariableOrigin` for more
    /// info.)
    pub(crate) origin: NllRegionVariableOrigin<'tcx>,

    /// Which universe is this region variable defined in? This is
    /// most often `ty::UniverseIndex::ROOT`, but when we encounter
    /// forall-quantifiers like `for<'a> { 'a = 'b }`, we would create
    /// the variable for `'a` in a fresh universe that extends ROOT.
    pub(crate) universe: ty::UniverseIndex,

    /// If this is 'static or an early-bound region, then this is
    /// `Some(X)` where `X` is the name of the region.
    pub(crate) external_name: Option<ty::Region<'tcx>>,
}

/// N.B., the variants in `Cause` are intentionally ordered. Lower
/// values are preferred when it comes to error messages. Do not
/// reorder willy nilly.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum Cause {
    /// point inserted because Local was live at the given Location
    LiveVar(Local, Location),

    /// point inserted because Local was dropped at the given Location
    DropVar(Local, Location),
}

/// A "type test" corresponds to an outlives constraint between a type
/// and a lifetime, like `T: 'x` or `<T as Foo>::Bar: 'x`. They are
/// translated from the `Verify` region constraints in the ordinary
/// inference context.
///
/// These sorts of constraints are handled differently than ordinary
/// constraints, at least at present. During type checking, the
/// `InferCtxt::process_registered_region_obligations` method will
/// attempt to convert a type test like `T: 'x` into an ordinary
/// outlives constraint when possible (for example, `&'a T: 'b` will
/// be converted into `'a: 'b` and registered as a `Constraint`).
///
/// In some cases, however, there are outlives relationships that are
/// not converted into a region constraint, but rather into one of
/// these "type tests". The distinction is that a type test does not
/// influence the inference result, but instead just examines the
/// values that we ultimately inferred for each region variable and
/// checks that they meet certain extra criteria. If not, an error
/// can be issued.
///
/// One reason for this is that these type tests typically boil down
/// to a check like `'a: 'x` where `'a` is a universally quantified
/// region -- and therefore not one whose value is really meant to be
/// *inferred*, precisely (this is not always the case: one can have a
/// type test like `<Foo as Trait<'?0>>::Bar: 'x`, where `'?0` is an
/// inference variable). Another reason is that these type tests can
/// involve *disjunction* -- that is, they can be satisfied in more
/// than one way.
///
/// For more information about this translation, see
/// `InferCtxt::process_registered_region_obligations` and
/// `InferCtxt::type_must_outlive` in `rustc_infer::infer::InferCtxt`.
#[derive(Clone)]
pub(crate) struct TypeTest<'tcx> {
    /// The type `T` that must outlive the region.
    pub generic_kind: GenericKind<'tcx>,

    /// The region `'x` that the type must outlive.
    pub lower_bound: RegionVid,

    /// The span to blame.
    pub span: Span,

    /// A test which, if met by the region `'x`, proves that this type
    /// constraint is satisfied.
    pub verify_bound: VerifyBound<'tcx>,
}

impl fmt::Debug for TypeTest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_bound(
            f: &mut fmt::Formatter<'_>,
            generic_kind: GenericKind<'_>,
            lower: RegionVid,
            bound: &VerifyBound<'_>,
        ) -> fmt::Result {
            let fmt_bounds =
                |f: &mut fmt::Formatter<'_>, bounds: &[VerifyBound<'_>]| -> fmt::Result {
                    let mut it = bounds.iter().peekable();
                    while let Some(bound) = it.next() {
                        fmt_bound(f, generic_kind, lower, bound)?;
                        if it.peek().is_some() {
                            write!(f, ", ")?
                        }
                    }
                    Ok(())
                };
            match bound {
                VerifyBound::IfEq(binder) => write!(f, "{:?} == {:?}", generic_kind, binder),
                VerifyBound::OutlivedBy(region) => write!(f, "{region:?}: {lower:?}"),
                VerifyBound::AnyBound(verify_bounds) => {
                    write!(f, "Any[")?;
                    fmt_bounds(f, verify_bounds)?;
                    write!(f, "]")
                }
                VerifyBound::AllBounds(verify_bounds) => {
                    write!(f, "All[")?;
                    fmt_bounds(f, verify_bounds)?;
                    write!(f, "]")
                }
                VerifyBound::IsEmpty => write!(f, "Empty({lower:?})"),
            }
        }
        write!(f, "TypeTest from {:?}[", self.span)?;
        fmt_bound(f, self.generic_kind, self.lower_bound, &self.verify_bound)?;
        write!(f, "] ⊢ {:?}: {:?}", self.generic_kind, self.lower_bound)
    }
}

/// When we have an unmet lifetime constraint, we try to propagate it outward (e.g. to a closure
/// environment). If we can't, it is an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionRelationCheckResult {
    Ok,
    Propagated,
    Error,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Trace<'a, 'tcx> {
    StartRegion,
    FromGraph(&'a OutlivesConstraint<'tcx>),
    FromStatic(RegionVid),
    NotVisited,
}

#[instrument(skip(infcx, sccs), level = "debug")]
fn sccs_info<'tcx>(infcx: &BorrowckInferCtxt<'tcx>, sccs: &ConstraintSccs) {
    use crate::renumber::RegionCtxt;

    let var_to_origin = infcx.reg_var_to_origin.borrow();

    let mut var_to_origin_sorted = var_to_origin.clone().into_iter().collect::<Vec<_>>();
    var_to_origin_sorted.sort_by_key(|vto| vto.0);

    if enabled!(Level::DEBUG) {
        let mut reg_vars_to_origins_str = "region variables to origins:\n".to_string();
        for (reg_var, origin) in var_to_origin_sorted.into_iter() {
            reg_vars_to_origins_str.push_str(&format!("{reg_var:?}: {origin:?}\n"));
        }
        debug!("{}", reg_vars_to_origins_str);
    }

    let num_components = sccs.num_sccs();
    let mut components = vec![FxIndexSet::default(); num_components];

    for (reg_var, scc_idx) in sccs.scc_indices().iter_enumerated() {
        let origin = var_to_origin.get(&reg_var).unwrap_or(&RegionCtxt::Unknown);
        components[scc_idx.as_usize()].insert((reg_var, *origin));
    }

    if enabled!(Level::DEBUG) {
        let mut components_str = "strongly connected components:".to_string();
        for (scc_idx, reg_vars_origins) in components.iter().enumerate() {
            let regions_info = reg_vars_origins.clone().into_iter().collect::<Vec<_>>();
            components_str.push_str(&format!(
                "{:?}: {:?},\n)",
                ConstraintSccIndex::from_usize(scc_idx),
                regions_info,
            ))
        }
        debug!("{}", components_str);
    }

    // calculate the best representative for each component
    let components_representatives = components
        .into_iter()
        .enumerate()
        .map(|(scc_idx, region_ctxts)| {
            let repr = region_ctxts
                .into_iter()
                .map(|reg_var_origin| reg_var_origin.1)
                .max_by(|x, y| x.preference_value().cmp(&y.preference_value()))
                .unwrap();

            (ConstraintSccIndex::from_usize(scc_idx), repr)
        })
        .collect::<FxIndexMap<_, _>>();

    let mut scc_node_to_edges = FxIndexMap::default();
    for (scc_idx, repr) in components_representatives.iter() {
        let edge_representatives = sccs
            .successors(*scc_idx)
            .iter()
            .map(|scc_idx| components_representatives[scc_idx])
            .collect::<Vec<_>>();
        scc_node_to_edges.insert((scc_idx, repr), edge_representatives);
    }

    debug!("SCC edges {:#?}", scc_node_to_edges);
}

#[derive(Clone, Debug)]
pub(crate) struct BestBlame<'tcx> {
    /// See docs on [`RegionInferenceContextInner::best_blame_constraint`] for what this is.
    path: Vec<OutlivesConstraint<'tcx>>,
    /// Index into `path` of the constraint most relevant to report to users.
    idx: usize,
}

impl<'tcx> BestBlame<'tcx> {
    pub(crate) fn to_obligation_cause(&self) -> ObligationCause<'tcx> {
        // FIXME - determine what we should do if we encounter multiple
        // `ConstraintCategory::Predicate` constraints. Currently, we just pick the first one.
        let cause_code = self
            .path
            .iter()
            .find_map(|constraint| {
                if let ConstraintCategory::Predicate(predicate_span) = constraint.category {
                    // We currently do not store the `DefId` in the `ConstraintCategory`
                    // for performances reasons. The error reporting code used by NLL only
                    // uses the span, so this doesn't cause any problems at the moment.
                    Some(ObligationCauseCode::WhereClause(CRATE_DEF_ID.to_def_id(), predicate_span))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ObligationCauseCode::Misc);

        ObligationCause::new(self.constraint().span, CRATE_DEF_ID, cause_code.clone())
    }

    pub(crate) fn constraint(&self) -> &OutlivesConstraint<'tcx> {
        &self.path[self.idx]
    }

    pub(crate) fn path(&self) -> &[OutlivesConstraint<'tcx>] {
        &self.path
    }
}
