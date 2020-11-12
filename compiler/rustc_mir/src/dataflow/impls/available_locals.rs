use std::{fmt, hash::BuildHasherDefault};

use rustc_data_structures::fx::FxIndexMap;
use rustc_index::{bit_set::BitSet, vec::Idx};
use rustc_middle::{
    mir::{self, Body, Local, Location},
    mir::{visit::Visitor, BorrowKind, Operand, Place, Rvalue, Statement, StatementKind},
};
use smallvec::SmallVec;

use crate::dataflow::{
    fmt::DebugWithAdapter, fmt::DebugWithContext, lattice, lattice::Dual, AnalysisDomain, Forward,
    GenKill, GenKillAnalysis,
};

/// Dataflow analysis for availability of locals. A local is available at a given program point,
/// if the value of the local can freely be used at the given program point.
/// Consider the following example:
/// ```
/// _1 = 4;
/// _2 = &_1;
/// _3 = *_2
/// ```
/// In the above example, `_2` is available at the third statement, so the statement can be
/// simplified to `_3 = _1`.
/// In general, an available local can be used freely on any path from the definition of `_2` to
/// statement `s`, if `_2` and its value is guaranteed to not be changed on all paths.
///
/// In the following example `_2` is not available in `bb2`,
/// since we do not know if `_2 = &5` is executed.
/// ```
/// bb0 {
///   _2 = &_1;
///   switchInt(_1) -> [4: bb1, otherwise: bb2]
/// }
///
/// bb1 {
///   _2 = &5;
/// }
///
/// bb2 {
///   _3 = *_2
/// }
/// ```
pub struct AvailableLocals {
    local_with_location_map: LocalWithLocationMap,
}

type State = Dual<BitSet<LocalWithLocationIndex>>;

impl<'a, 'tcx> AvailableLocals {
    pub fn new(body: &'a Body<'tcx>) -> Self {
        let local_with_location_map = LocalWithLocationMap::new(body);
        debug!("Constructed {:?}", local_with_location_map);
        Self { local_with_location_map }
    }

    /// Returns the state with context attached, making debug easier
    pub fn debug_state(&'a self, state: &'a State) -> impl fmt::Debug + 'a {
        DebugWithAdapter { this: state, ctxt: &self.local_with_location_map }
    }

    /// Returns whether the given local is available at the given state
    pub fn is_available(&self, local: Local, state: &'a State) -> bool {
        let locations_opt = self.local_with_location_map.local_with_locations(local);
        if let Some(mut locations) = locations_opt {
            locations.any(|place_taken_ref_local_with_location| {
                state.0.contains(place_taken_ref_local_with_location)
            })
        } else {
            false
        }
    }

    fn transfer_function<T>(&'a self, available: &'a mut T) -> TransferFunction<'a, T>
    where
        T: GenKill<LocalWithLocationIndex>,
    {
        TransferFunction { available, local_with_location_map: &self.local_with_location_map }
    }
}
impl AnalysisDomain<'tcx> for AvailableLocals {
    /// We want a *forward must* analysis.
    /// At each join point, we only keep locals that were available in both basic blocks.
    /// For this, we use the Dual of a semi-lattice, such that intersection is the join operator
    type Domain = State;
    type Direction = Forward;

    const NAME: &'static str = "available_locals";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // If we were not using the dual, we would have nothing available, but since we are using
        // the dual, we also "reverse" the bottom to be filled
        // bottom = nothing is available
        let bottom = BitSet::new_filled(self.local_with_location_map.len());
        lattice::Dual(bottom)
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        // In the start block, nothing is available at first
        state.0.clear();

        // only the function parameters are available
        for arg in body.args_iter() {
            state.0.insert(self.local_with_location_map.local_with_specific_location(arg, None));
        }
    }
}

impl GenKillAnalysis<'tcx> for AvailableLocals {
    type Idx = LocalWithLocationIndex;

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_terminator(terminator, location);
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        _dest_place: mir::Place<'tcx>,
    ) {
        // Conservatively do not try to reason about calls
    }
}

// Compiling rustc stage2 reveals that 95% of locals are only used at 2 locations
const LOCAL_WITH_LOCATION_SMALLVEC_CAP: usize = 2;

#[derive(Debug)]
struct LocalWithLocationMap(
    FxIndexMap<
        Local,
        SmallVec<[(LocalWithLocationIndex, Option<Location>); LOCAL_WITH_LOCATION_SMALLVEC_CAP]>,
    >,
);

impl LocalWithLocationMap {
    fn new(body: &Body<'tcx>) -> Self {
        let mut map = FxIndexMap::with_capacity_and_hasher(
            1 + body.arg_count + body.vars_and_temps_iter().len(),
            BuildHasherDefault::default(),
        );

        // return pointer
        map.entry(body.local_decls.indices().next().unwrap())
            .or_insert(SmallVec::with_capacity(2))
            .push((0usize.into(), None));

        for arg in body.args_iter() {
            map.entry(arg).or_insert(SmallVec::with_capacity(2)).push((0usize.into(), None));
        }

        let mut builder = LocalWithLocationMapBuilder(&mut map);
        builder.visit_body(body);

        // set indices
        let mut idx = LocalWithLocationIndex::from(0usize);
        for x in map.values_mut() {
            for (idx_to_set, _) in x.iter_mut() {
                *idx_to_set = idx;
                idx.increment_by(1);
            }
        }

        Self(map)
    }

    fn local_with_specific_location(
        &self,
        local: Local,
        location: Option<Location>,
    ) -> LocalWithLocationIndex {
        debug!("Looking for {:?} in {:?}", (local, location), self.0);
        let locations = self.0.get(&local).unwrap();
        *locations
            .iter()
            .find(|(_, l)| *l == location)
            .map(|(location_idx, _)| location_idx)
            .unwrap()
    }

    fn local_with_locations(
        &self,
        local: Local,
    ) -> Option<impl Iterator<Item = LocalWithLocationIndex> + '_> {
        debug!("Looking for {:?} in {:?}", local, self.0);
        let locations = self.0.get(&local)?;
        return Some(locations.iter().map(move |(location_idx, _)| *location_idx));
    }

    fn len(&self) -> usize {
        let mut count = 0;
        for v in self.0.values() {
            count += v.len();
        }
        count
    }
}

struct LocalWithLocationMapBuilder<'a>(
    &'a mut FxIndexMap<Local, SmallVec<[(LocalWithLocationIndex, Option<Location>); 2]>>,
);

impl<'a, 'tcx> Visitor<'tcx> for LocalWithLocationMapBuilder<'a> {
    fn visit_assign(&mut self, place: &Place<'tcx>, _: &Rvalue<'tcx>, location: Location) {
        self.0
            .entry(place.local)
            .or_insert(SmallVec::with_capacity(2))
            .push((0usize.into(), Some(location)));
    }
}

rustc_index::newtype_index! {
    pub struct LocalWithLocationIndex {
        DEBUG_FORMAT = "ll_{}"
    }
}

impl DebugWithContext<AvailableLocals> for LocalWithLocationIndex {
    fn fmt_with(&self, ctx: &AvailableLocals, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DebugWithContext::fmt_with(self, &ctx.local_with_location_map, f)
    }
}

impl DebugWithContext<LocalWithLocationMap> for LocalWithLocationIndex {
    fn fmt_with(&self, ctx: &LocalWithLocationMap, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // find local and location
        for (local, value) in ctx.0.iter() {
            if let Some((idx, location)) = value.iter().find(|(idx, _)| idx == self) {
                return write!(f, "{:?}: {:?}", idx, (local, location));
            }
        }
        unreachable!()
    }
}

struct TransferFunction<'a, T>
where
    T: GenKill<LocalWithLocationIndex>,
{
    available: &'a mut T,
    local_with_location_map: &'a LocalWithLocationMap,
}

impl<'a, 'tcx, T> TransferFunction<'a, T>
where
    T: GenKill<LocalWithLocationIndex>,
{
    fn invalidate_local(&mut self, local_invalidated: Local, _location: Location) {
        if let Some(locations) =
            self.local_with_location_map.local_with_locations(local_invalidated)
        {
            for x in locations {
                debug!("Invalidating {:?} which corresponds to {:?}", x, local_invalidated);
                self.available.kill(x);
            }
        }
    }

    fn add_available(&mut self, local: Local, rvalue: Rvalue<'tcx>, location: Location) {
        self.invalidate_local(local, location);

        // If rvalue is a move into the assigned local, then the local being moved should be invalidated
        match rvalue {
            Rvalue::Use(Operand::Move(moved)) => self.invalidate_local(moved.local, location),
            _ => {}
        }

        let all_participating_alive =
            rvalue.participating_locals(location).all(|participating_local| {
                let location_opt =
                    self.local_with_location_map.local_with_locations(participating_local);
                if let Some(mut locations) = location_opt {
                    locations.any(|participating_local_idx| {
                        self.available.is_alive(participating_local_idx)
                    })
                } else {
                    false
                }
            });
        if all_participating_alive {
            let index =
                self.local_with_location_map.local_with_specific_location(local, Some(location));
            debug!("Inserting {:?} which corresponds to {:?}", index, (local, Some(location)));
            self.available.gen(index);
        }
    }
}

impl<'a, 'tcx, T> Visitor<'tcx> for TransferFunction<'a, T>
where
    T: GenKill<LocalWithLocationIndex>,
{
    fn visit_assign(
        &mut self,
        assigned_place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location,
    ) {
        if let Some(assigned_local) = assigned_place.as_local() {
            let disallowed_place_opt = match rvalue {
                Rvalue::Ref(_, BorrowKind::Mut { .. } | BorrowKind::Shallow, ref_of) => {
                    Some(ref_of)
                }
                Rvalue::AddressOf(_, ref_of) => Some(ref_of),
                _ => None,
            };

            if let Some(disallowed_place) = disallowed_place_opt {
                // It is difficult to reason about availability of mutable places or raw pointers, so throw out any
                // availability information about the local taken a reference of.
                debug!(
                    "Found disallowed reference of `{:?}`. Invalidating {:?} and {:?}",
                    disallowed_place, disallowed_place.local, assigned_local
                );
                self.invalidate_local(disallowed_place.local, location);
                self.invalidate_local(assigned_local, location)
            } else {
                self.add_available(assigned_local, rvalue.clone(), location);
            }
        }

        self.super_assign(assigned_place, rvalue, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::StorageDead(dead) => {
                self.invalidate_local(dead, location);
            }
            _ => self.super_statement(statement, location),
        }
    }
}
