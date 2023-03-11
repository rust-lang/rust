//! This module provides a framework on top of the normal MIR dataflow framework to simplify the
//! implementation of analyses that track information about the values stored in certain places.
//! We are using the term "place" here to refer to a `mir::Place` (a place expression) instead of
//! an `interpret::Place` (a memory location).
//!
//! The default methods of [`ValueAnalysis`] (prefixed with `super_` instead of `handle_`)
//! provide some behavior that should be valid for all abstract domains that are based only on the
//! value stored in a certain place. On top of these default rules, an implementation should
//! override some of the `handle_` methods. For an example, see `ConstAnalysis`.
//!
//! An implementation must also provide a [`Map`]. Before the analysis begins, all places that
//! should be tracked during the analysis must be registered. During the analysis, no new places
//! can be registered. The [`State`] can be queried to retrieve the abstract value stored for a
//! certain place by passing the map.
//!
//! This framework is currently experimental. Originally, it supported shared references and enum
//! variants. However, it was discovered that both of these were unsound, and especially references
//! had subtle but serious issues. In the future, they could be added back in, but we should clarify
//! the rules for optimizations that rely on the aliasing model first.
//!
//!
//! # Notes
//!
//! - The bottom state denotes uninitialized memory. Because we are only doing a sound approximation
//! of the actual execution, we can also use this state for places where access would be UB.
//!
//! - The assignment logic in `State::insert_place_idx` assumes that the places are non-overlapping,
//! or identical. Note that this refers to place expressions, not memory locations.
//!
//! - Currently, places that have their reference taken cannot be tracked. Although this would be
//! possible, it has to rely on some aliasing model, which we are not ready to commit to yet.
//! Because of that, we can assume that the only way to change the value behind a tracked place is
//! by direct assignment.

use std::fmt::{Debug, Formatter};

use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;

use crate::lattice::{HasBottom, HasTop};
use crate::{
    fmt::DebugWithContext, Analysis, AnalysisDomain, CallReturnPlaces, JoinSemiLattice,
    SwitchIntEdgeEffects,
};

pub trait ValueAnalysis<'tcx> {
    /// For each place of interest, the analysis tracks a value of the given type.
    type Value: Clone + JoinSemiLattice + HasBottom + HasTop;

    const NAME: &'static str;

    fn map(&self) -> &Map;

    fn handle_statement(&self, statement: &Statement<'tcx>, state: &mut State<Self::Value>) {
        self.super_statement(statement, state)
    }

    fn super_statement(&self, statement: &Statement<'tcx>, state: &mut State<Self::Value>) {
        match &statement.kind {
            StatementKind::Assign(box (place, rvalue)) => {
                self.handle_assign(*place, rvalue, state);
            }
            StatementKind::SetDiscriminant { box ref place, .. } => {
                state.flood_discr(place.as_ref(), self.map());
            }
            StatementKind::Intrinsic(box intrinsic) => {
                self.handle_intrinsic(intrinsic, state);
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                // StorageLive leaves the local in an uninitialized state.
                // StorageDead makes it UB to access the local afterwards.
                state.flood_with(Place::from(*local).as_ref(), self.map(), Self::Value::bottom());
            }
            StatementKind::Deinit(box place) => {
                // Deinit makes the place uninitialized.
                state.flood_with(place.as_ref(), self.map(), Self::Value::bottom());
            }
            StatementKind::Retag(..) => {
                // We don't track references.
            }
            StatementKind::ConstEvalCounter
            | StatementKind::Nop
            | StatementKind::FakeRead(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::AscribeUserType(..) => (),
        }
    }

    fn handle_intrinsic(
        &self,
        intrinsic: &NonDivergingIntrinsic<'tcx>,
        state: &mut State<Self::Value>,
    ) {
        self.super_intrinsic(intrinsic, state);
    }

    fn super_intrinsic(
        &self,
        intrinsic: &NonDivergingIntrinsic<'tcx>,
        state: &mut State<Self::Value>,
    ) {
        match intrinsic {
            NonDivergingIntrinsic::Assume(..) => {
                // Could use this, but ignoring it is sound.
            }
            NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping { dst, .. }) => {
                if let Some(place) = dst.place() {
                    state.flood(place.as_ref(), self.map());
                }
            }
        }
    }

    fn handle_assign(
        &self,
        target: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) {
        self.super_assign(target, rvalue, state)
    }

    fn super_assign(
        &self,
        target: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) {
        let result = self.handle_rvalue(rvalue, state);
        state.assign(target.as_ref(), result, self.map());
    }

    fn handle_rvalue(
        &self,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) -> ValueOrPlace<Self::Value> {
        self.super_rvalue(rvalue, state)
    }

    fn super_rvalue(
        &self,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) -> ValueOrPlace<Self::Value> {
        match rvalue {
            Rvalue::Use(operand) => self.handle_operand(operand, state),
            Rvalue::CopyForDeref(place) => self.handle_operand(&Operand::Copy(*place), state),
            Rvalue::Ref(..) | Rvalue::AddressOf(..) => {
                // We don't track such places.
                ValueOrPlace::top()
            }
            Rvalue::Repeat(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Aggregate(..)
            | Rvalue::ShallowInitBox(..) => {
                // No modification is possible through these r-values.
                ValueOrPlace::top()
            }
        }
    }

    fn handle_operand(
        &self,
        operand: &Operand<'tcx>,
        state: &mut State<Self::Value>,
    ) -> ValueOrPlace<Self::Value> {
        self.super_operand(operand, state)
    }

    fn super_operand(
        &self,
        operand: &Operand<'tcx>,
        state: &mut State<Self::Value>,
    ) -> ValueOrPlace<Self::Value> {
        match operand {
            Operand::Constant(box constant) => {
                ValueOrPlace::Value(self.handle_constant(constant, state))
            }
            Operand::Copy(place) | Operand::Move(place) => {
                // On move, we would ideally flood the place with bottom. But with the current
                // framework this is not possible (similar to `InterpCx::eval_operand`).
                self.map()
                    .find(place.as_ref())
                    .map(ValueOrPlace::Place)
                    .unwrap_or(ValueOrPlace::top())
            }
        }
    }

    fn handle_constant(
        &self,
        constant: &Constant<'tcx>,
        state: &mut State<Self::Value>,
    ) -> Self::Value {
        self.super_constant(constant, state)
    }

    fn super_constant(
        &self,
        _constant: &Constant<'tcx>,
        _state: &mut State<Self::Value>,
    ) -> Self::Value {
        Self::Value::top()
    }

    /// The effect of a successful function call return should not be
    /// applied here, see [`Analysis::apply_terminator_effect`].
    fn handle_terminator(&self, terminator: &Terminator<'tcx>, state: &mut State<Self::Value>) {
        self.super_terminator(terminator, state)
    }

    fn super_terminator(&self, terminator: &Terminator<'tcx>, state: &mut State<Self::Value>) {
        match &terminator.kind {
            TerminatorKind::Call { .. } | TerminatorKind::InlineAsm { .. } => {
                // Effect is applied by `handle_call_return`.
            }
            TerminatorKind::Drop { place, .. } => {
                state.flood_with(place.as_ref(), self.map(), Self::Value::bottom());
            }
            TerminatorKind::Yield { .. } => {
                // They would have an effect, but are not allowed in this phase.
                bug!("encountered disallowed terminator");
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Assert { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                // These terminators have no effect on the analysis.
            }
        }
    }

    fn handle_call_return(
        &self,
        return_places: CallReturnPlaces<'_, 'tcx>,
        state: &mut State<Self::Value>,
    ) {
        self.super_call_return(return_places, state)
    }

    fn super_call_return(
        &self,
        return_places: CallReturnPlaces<'_, 'tcx>,
        state: &mut State<Self::Value>,
    ) {
        return_places.for_each(|place| {
            state.flood(place.as_ref(), self.map());
        })
    }

    fn handle_switch_int(
        &self,
        discr: &Operand<'tcx>,
        apply_edge_effects: &mut impl SwitchIntEdgeEffects<State<Self::Value>>,
    ) {
        self.super_switch_int(discr, apply_edge_effects)
    }

    fn super_switch_int(
        &self,
        _discr: &Operand<'tcx>,
        _apply_edge_effects: &mut impl SwitchIntEdgeEffects<State<Self::Value>>,
    ) {
    }

    fn wrap(self) -> ValueAnalysisWrapper<Self>
    where
        Self: Sized,
    {
        ValueAnalysisWrapper(self)
    }
}

pub struct ValueAnalysisWrapper<T>(pub T);

impl<'tcx, T: ValueAnalysis<'tcx>> AnalysisDomain<'tcx> for ValueAnalysisWrapper<T> {
    type Domain = State<T::Value>;

    type Direction = crate::Forward;

    const NAME: &'static str = T::NAME;

    fn bottom_value(&self, _body: &Body<'tcx>) -> Self::Domain {
        State(StateData::Unreachable)
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        // The initial state maps all tracked places of argument projections to ⊤ and the rest to ⊥.
        assert!(matches!(state.0, StateData::Unreachable));
        let values = IndexVec::from_elem_n(T::Value::bottom(), self.0.map().value_count);
        *state = State(StateData::Reachable(values));
        for arg in body.args_iter() {
            state.flood(PlaceRef { local: arg, projection: &[] }, self.0.map());
        }
    }
}

impl<'tcx, T> Analysis<'tcx> for ValueAnalysisWrapper<T>
where
    T: ValueAnalysis<'tcx>,
{
    fn apply_statement_effect(
        &self,
        state: &mut Self::Domain,
        statement: &Statement<'tcx>,
        _location: Location,
    ) {
        if state.is_reachable() {
            self.0.handle_statement(statement, state);
        }
    }

    fn apply_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        _location: Location,
    ) {
        if state.is_reachable() {
            self.0.handle_terminator(terminator, state);
        }
    }

    fn apply_call_return_effect(
        &self,
        state: &mut Self::Domain,
        _block: BasicBlock,
        return_places: crate::CallReturnPlaces<'_, 'tcx>,
    ) {
        if state.is_reachable() {
            self.0.handle_call_return(return_places, state)
        }
    }

    fn apply_switch_int_edge_effects(
        &self,
        _block: BasicBlock,
        discr: &Operand<'tcx>,
        apply_edge_effects: &mut impl SwitchIntEdgeEffects<Self::Domain>,
    ) {
        // FIXME: Dataflow framework provides no access to current state here.
        self.0.handle_switch_int(discr, apply_edge_effects)
    }
}

rustc_index::newtype_index!(
    /// This index uniquely identifies a place.
    ///
    /// Not every place has a `PlaceIndex`, and not every `PlaceIndex` correspondends to a tracked
    /// place. However, every tracked place and all places along its projection have a `PlaceIndex`.
    pub struct PlaceIndex {}
);

rustc_index::newtype_index!(
    /// This index uniquely identifies a tracked place and therefore a slot in [`State`].
    ///
    /// It is an implementation detail of this module.
    struct ValueIndex {}
);

/// See [`State`].
#[derive(PartialEq, Eq, Debug)]
enum StateData<V> {
    Reachable(IndexVec<ValueIndex, V>),
    Unreachable,
}

impl<V: Clone> Clone for StateData<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Reachable(x) => Self::Reachable(x.clone()),
            Self::Unreachable => Self::Unreachable,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (&mut *self, source) {
            (Self::Reachable(x), Self::Reachable(y)) => {
                // We go through `raw` here, because `IndexVec` currently has a naive `clone_from`.
                x.raw.clone_from(&y.raw);
            }
            _ => *self = source.clone(),
        }
    }
}

/// The dataflow state for an instance of [`ValueAnalysis`].
///
/// Every instance specifies a lattice that represents the possible values of a single tracked
/// place. If we call this lattice `V` and set of tracked places `P`, then a [`State`] is an
/// element of `{unreachable} ∪ (P -> V)`. This again forms a lattice, where the bottom element is
/// `unreachable` and the top element is the mapping `p ↦ ⊤`. Note that the mapping `p ↦ ⊥` is not
/// the bottom element (because joining an unreachable and any other reachable state yields a
/// reachable state). All operations on unreachable states are ignored.
///
/// Flooding means assigning a value (by default `⊤`) to all tracked projections of a given place.
#[derive(PartialEq, Eq, Debug)]
pub struct State<V>(StateData<V>);

impl<V: Clone> Clone for State<V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0);
    }
}

impl<V: Clone + HasTop + HasBottom> State<V> {
    pub fn is_reachable(&self) -> bool {
        matches!(&self.0, StateData::Reachable(_))
    }

    pub fn mark_unreachable(&mut self) {
        self.0 = StateData::Unreachable;
    }

    pub fn flood_all(&mut self) {
        self.flood_all_with(V::top())
    }

    pub fn flood_all_with(&mut self, value: V) {
        let StateData::Reachable(values) = &mut self.0 else { return };
        values.raw.fill(value);
    }

    pub fn flood_with(&mut self, place: PlaceRef<'_>, map: &Map, value: V) {
        let StateData::Reachable(values) = &mut self.0 else { return };
        map.for_each_aliasing_place(place, None, &mut |place| {
            if let Some(vi) = map.places[place].value_index {
                values[vi] = value.clone();
            }
        });
    }

    pub fn flood(&mut self, place: PlaceRef<'_>, map: &Map) {
        self.flood_with(place, map, V::top())
    }

    pub fn flood_discr_with(&mut self, place: PlaceRef<'_>, map: &Map, value: V) {
        let StateData::Reachable(values) = &mut self.0 else { return };
        map.for_each_aliasing_place(place, Some(TrackElem::Discriminant), &mut |place| {
            if let Some(vi) = map.places[place].value_index {
                values[vi] = value.clone();
            }
        });
    }

    pub fn flood_discr(&mut self, place: PlaceRef<'_>, map: &Map) {
        self.flood_discr_with(place, map, V::top())
    }

    /// Low-level method that assigns to a place.
    /// This does nothing if the place is not tracked.
    ///
    /// The target place must have been flooded before calling this method.
    pub fn insert_idx(&mut self, target: PlaceIndex, result: ValueOrPlace<V>, map: &Map) {
        match result {
            ValueOrPlace::Value(value) => self.insert_value_idx(target, value, map),
            ValueOrPlace::Place(source) => self.insert_place_idx(target, source, map),
        }
    }

    /// Low-level method that assigns a value to a place.
    /// This does nothing if the place is not tracked.
    ///
    /// The target place must have been flooded before calling this method.
    pub fn insert_value_idx(&mut self, target: PlaceIndex, value: V, map: &Map) {
        let StateData::Reachable(values) = &mut self.0 else { return };
        if let Some(value_index) = map.places[target].value_index {
            values[value_index] = value;
        }
    }

    /// Copies `source` to `target`, including all tracked places beneath.
    ///
    /// If `target` contains a place that is not contained in `source`, it will be overwritten with
    /// Top. Also, because this will copy all entries one after another, it may only be used for
    /// places that are non-overlapping or identical.
    ///
    /// The target place must have been flooded before calling this method.
    fn insert_place_idx(&mut self, target: PlaceIndex, source: PlaceIndex, map: &Map) {
        let StateData::Reachable(values) = &mut self.0 else { return };

        // If both places are tracked, we copy the value to the target.
        // If the target is tracked, but the source is not, we do nothing, as invalidation has
        // already been performed.
        if let Some(target_value) = map.places[target].value_index {
            if let Some(source_value) = map.places[source].value_index {
                values[target_value] = values[source_value].clone();
            }
        }
        for target_child in map.children(target) {
            // Try to find corresponding child and recurse. Reasoning is similar as above.
            let projection = map.places[target_child].proj_elem.unwrap();
            if let Some(source_child) = map.projections.get(&(source, projection)) {
                self.insert_place_idx(target_child, *source_child, map);
            }
        }
    }

    /// Helper method to interpret `target = result`.
    pub fn assign(&mut self, target: PlaceRef<'_>, result: ValueOrPlace<V>, map: &Map) {
        self.flood(target, map);
        if let Some(target) = map.find(target) {
            self.insert_idx(target, result, map);
        }
    }

    /// Helper method for assignments to a discriminant.
    pub fn assign_discr(&mut self, target: PlaceRef<'_>, result: ValueOrPlace<V>, map: &Map) {
        self.flood_discr(target, map);
        if let Some(target) = map.find_discr(target) {
            self.insert_idx(target, result, map);
        }
    }

    /// Retrieve the value stored for a place, or ⊤ if it is not tracked.
    pub fn get(&self, place: PlaceRef<'_>, map: &Map) -> V {
        map.find(place).map(|place| self.get_idx(place, map)).unwrap_or(V::top())
    }

    /// Retrieve the value stored for a place, or ⊤ if it is not tracked.
    pub fn get_discr(&self, place: PlaceRef<'_>, map: &Map) -> V {
        match map.find_discr(place) {
            Some(place) => self.get_idx(place, map),
            None => V::top(),
        }
    }

    /// Retrieve the value stored for a place index, or ⊤ if it is not tracked.
    pub fn get_idx(&self, place: PlaceIndex, map: &Map) -> V {
        match &self.0 {
            StateData::Reachable(values) => {
                map.places[place].value_index.map(|v| values[v].clone()).unwrap_or(V::top())
            }
            StateData::Unreachable => {
                // Because this is unreachable, we can return any value we want.
                V::bottom()
            }
        }
    }
}

impl<V: JoinSemiLattice + Clone> JoinSemiLattice for State<V> {
    fn join(&mut self, other: &Self) -> bool {
        match (&mut self.0, &other.0) {
            (_, StateData::Unreachable) => false,
            (StateData::Unreachable, _) => {
                *self = other.clone();
                true
            }
            (StateData::Reachable(this), StateData::Reachable(other)) => this.join(other),
        }
    }
}

/// Partial mapping from [`Place`] to [`PlaceIndex`], where some places also have a [`ValueIndex`].
///
/// This data structure essentially maintains a tree of places and their projections. Some
/// additional bookkeeping is done, to speed up traversal over this tree:
/// - For iteration, every [`PlaceInfo`] contains an intrusive linked list of its children.
/// - To directly get the child for a specific projection, there is a `projections` map.
#[derive(Debug)]
pub struct Map {
    locals: IndexVec<Local, Option<PlaceIndex>>,
    projections: FxHashMap<(PlaceIndex, TrackElem), PlaceIndex>,
    places: IndexVec<PlaceIndex, PlaceInfo>,
    value_count: usize,
}

impl Map {
    fn new() -> Self {
        Self {
            locals: IndexVec::new(),
            projections: FxHashMap::default(),
            places: IndexVec::new(),
            value_count: 0,
        }
    }

    /// Returns a map that only tracks places whose type passes the filter.
    ///
    /// This is currently the only way to create a [`Map`]. The way in which the tracked places are
    /// chosen is an implementation detail and may not be relied upon (other than that their type
    /// passes the filter).
    pub fn from_filter<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        filter: impl FnMut(Ty<'tcx>) -> bool,
        place_limit: Option<usize>,
    ) -> Self {
        let mut map = Self::new();
        let exclude = excluded_locals(body);
        map.register_with_filter(tcx, body, filter, exclude, place_limit);
        debug!("registered {} places ({} nodes in total)", map.value_count, map.places.len());
        map
    }

    /// Register all non-excluded places that pass the filter.
    fn register_with_filter<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        mut filter: impl FnMut(Ty<'tcx>) -> bool,
        exclude: BitSet<Local>,
        place_limit: Option<usize>,
    ) {
        // We use this vector as stack, pushing and popping projections.
        let mut projection = Vec::new();
        for (local, decl) in body.local_decls.iter_enumerated() {
            if !exclude.contains(local) {
                self.register_with_filter_rec(
                    tcx,
                    local,
                    &mut projection,
                    decl.ty,
                    &mut filter,
                    place_limit,
                );
            }
        }
    }

    /// Potentially register the (local, projection) place and its fields, recursively.
    ///
    /// Invariant: The projection must only contain trackable elements.
    fn register_with_filter_rec<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        local: Local,
        projection: &mut Vec<PlaceElem<'tcx>>,
        ty: Ty<'tcx>,
        filter: &mut impl FnMut(Ty<'tcx>) -> bool,
        place_limit: Option<usize>,
    ) {
        if let Some(place_limit) = place_limit && self.value_count >= place_limit {
            return
        }

        // We know that the projection only contains trackable elements.
        let place = self.make_place(local, projection).unwrap();

        // Allocate a value slot if it doesn't have one, and the user requested one.
        if self.places[place].value_index.is_none() && filter(ty) {
            self.places[place].value_index = Some(self.value_count.into());
            self.value_count += 1;
        }

        if ty.is_enum() {
            let discr_ty = ty.discriminant_ty(tcx);
            if filter(discr_ty) {
                let discr = *self
                    .projections
                    .entry((place, TrackElem::Discriminant))
                    .or_insert_with(|| {
                        // Prepend new child to the linked list.
                        let next = self.places.push(PlaceInfo::new(Some(TrackElem::Discriminant)));
                        self.places[next].next_sibling = self.places[place].first_child;
                        self.places[place].first_child = Some(next);
                        next
                    });

                // Allocate a value slot if it doesn't have one.
                if self.places[discr].value_index.is_none() {
                    self.places[discr].value_index = Some(self.value_count.into());
                    self.value_count += 1;
                }
            }
        }

        // Recurse with all fields of this place.
        iter_fields(ty, tcx, ty::ParamEnv::reveal_all(), |variant, field, ty| {
            if let Some(variant) = variant {
                projection.push(PlaceElem::Downcast(None, variant));
                let _ = self.make_place(local, projection);
                projection.push(PlaceElem::Field(field, ty));
                self.register_with_filter_rec(tcx, local, projection, ty, filter, place_limit);
                projection.pop();
                projection.pop();
                return;
            }
            projection.push(PlaceElem::Field(field, ty));
            self.register_with_filter_rec(tcx, local, projection, ty, filter, place_limit);
            projection.pop();
        });
    }

    /// Tries to add the place to the map, without allocating a value slot.
    ///
    /// Can fail if the projection contains non-trackable elements.
    fn make_place<'tcx>(
        &mut self,
        local: Local,
        projection: &[PlaceElem<'tcx>],
    ) -> Result<PlaceIndex, ()> {
        // Get the base index of the local.
        let mut index =
            *self.locals.get_or_insert_with(local, || self.places.push(PlaceInfo::new(None)));

        // Apply the projection.
        for &elem in projection {
            let elem = elem.try_into()?;
            index = *self.projections.entry((index, elem)).or_insert_with(|| {
                // Prepend new child to the linked list.
                let next = self.places.push(PlaceInfo::new(Some(elem)));
                self.places[next].next_sibling = self.places[index].first_child;
                self.places[index].first_child = Some(next);
                next
            });
        }

        Ok(index)
    }

    /// Returns the number of tracked places, i.e., those for which a value can be stored.
    pub fn tracked_places(&self) -> usize {
        self.value_count
    }

    /// Applies a single projection element, yielding the corresponding child.
    pub fn apply(&self, place: PlaceIndex, elem: TrackElem) -> Option<PlaceIndex> {
        self.projections.get(&(place, elem)).copied()
    }

    /// Locates the given place, if it exists in the tree.
    pub fn find_extra(
        &self,
        place: PlaceRef<'_>,
        extra: impl IntoIterator<Item = TrackElem>,
    ) -> Option<PlaceIndex> {
        let mut index = *self.locals.get(place.local)?.as_ref()?;

        for &elem in place.projection {
            index = self.apply(index, elem.try_into().ok()?)?;
        }
        for elem in extra {
            index = self.apply(index, elem)?;
        }

        Some(index)
    }

    /// Locates the given place, if it exists in the tree.
    pub fn find(&self, place: PlaceRef<'_>) -> Option<PlaceIndex> {
        self.find_extra(place, [])
    }

    /// Locates the given place and applies `Discriminant`, if it exists in the tree.
    pub fn find_discr(&self, place: PlaceRef<'_>) -> Option<PlaceIndex> {
        self.find_extra(place, [TrackElem::Discriminant])
    }

    /// Iterate over all direct children.
    pub fn children(&self, parent: PlaceIndex) -> impl Iterator<Item = PlaceIndex> + '_ {
        Children::new(self, parent)
    }

    /// Invoke a function on the given place and all places that may alias it.
    ///
    /// In particular, when the given place has a variant downcast, we invoke the function on all
    /// the other variants.
    ///
    /// `tail_elem` allows to support discriminants that are not a place in MIR, but that we track
    /// as such.
    pub fn for_each_aliasing_place(
        &self,
        place: PlaceRef<'_>,
        tail_elem: Option<TrackElem>,
        f: &mut impl FnMut(PlaceIndex),
    ) {
        if place.is_indirect() {
            // We do not track indirect places.
            return;
        }
        let Some(&Some(mut index)) = self.locals.get(place.local) else {
            // The local is not tracked at all, so it does not alias anything.
            return;
        };
        let elems = place
            .projection
            .iter()
            .map(|&elem| elem.try_into())
            .chain(tail_elem.map(Ok).into_iter());
        for elem in elems {
            // A field aliases the parent place.
            f(index);

            let Ok(elem) = elem else { return };
            let sub = self.apply(index, elem);
            if let TrackElem::Variant(..) | TrackElem::Discriminant = elem {
                // Enum variant fields and enum discriminants alias each another.
                self.for_each_variant_sibling(index, sub, f);
            }
            if let Some(sub) = sub {
                index = sub
            } else {
                return;
            }
        }
        self.preorder_invoke(index, f);
    }

    /// Invoke the given function on all the descendants of the given place, except one branch.
    fn for_each_variant_sibling(
        &self,
        parent: PlaceIndex,
        preserved_child: Option<PlaceIndex>,
        f: &mut impl FnMut(PlaceIndex),
    ) {
        for sibling in self.children(parent) {
            let elem = self.places[sibling].proj_elem;
            // Only invalidate variants and discriminant. Fields (for generators) are not
            // invalidated by assignment to a variant.
            if let Some(TrackElem::Variant(..) | TrackElem::Discriminant) = elem
                // Only invalidate the other variants, the current one is fine.
                && Some(sibling) != preserved_child
            {
                self.preorder_invoke(sibling, f);
            }
        }
    }

    /// Invoke a function on the given place and all descendants.
    fn preorder_invoke(&self, root: PlaceIndex, f: &mut impl FnMut(PlaceIndex)) {
        f(root);
        for child in self.children(root) {
            self.preorder_invoke(child, f);
        }
    }
}

/// This is the information tracked for every [`PlaceIndex`] and is stored by [`Map`].
///
/// Together, `first_child` and `next_sibling` form an intrusive linked list, which is used to
/// model a tree structure (a replacement for a member like `children: Vec<PlaceIndex>`).
#[derive(Debug)]
struct PlaceInfo {
    /// We store a [`ValueIndex`] if and only if the placed is tracked by the analysis.
    value_index: Option<ValueIndex>,

    /// The projection used to go from parent to this node (only None for root).
    proj_elem: Option<TrackElem>,

    /// The left-most child.
    first_child: Option<PlaceIndex>,

    /// Index of the sibling to the right of this node.
    next_sibling: Option<PlaceIndex>,
}

impl PlaceInfo {
    fn new(proj_elem: Option<TrackElem>) -> Self {
        Self { next_sibling: None, first_child: None, proj_elem, value_index: None }
    }
}

struct Children<'a> {
    map: &'a Map,
    next: Option<PlaceIndex>,
}

impl<'a> Children<'a> {
    fn new(map: &'a Map, parent: PlaceIndex) -> Self {
        Self { map, next: map.places[parent].first_child }
    }
}

impl<'a> Iterator for Children<'a> {
    type Item = PlaceIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            Some(child) => {
                self.next = self.map.places[child].next_sibling;
                Some(child)
            }
            None => None,
        }
    }
}

/// Used as the result of an operand or r-value.
#[derive(Debug)]
pub enum ValueOrPlace<V> {
    Value(V),
    Place(PlaceIndex),
}

impl<V: HasTop> ValueOrPlace<V> {
    pub fn top() -> Self {
        ValueOrPlace::Value(V::top())
    }
}

/// The set of projection elements that can be used by a tracked place.
///
/// Although only field projections are currently allowed, this could change in the future.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackElem {
    Field(Field),
    Variant(VariantIdx),
    Discriminant,
}

impl<V, T> TryFrom<ProjectionElem<V, T>> for TrackElem {
    type Error = ();

    fn try_from(value: ProjectionElem<V, T>) -> Result<Self, Self::Error> {
        match value {
            ProjectionElem::Field(field, _) => Ok(TrackElem::Field(field)),
            ProjectionElem::Downcast(_, idx) => Ok(TrackElem::Variant(idx)),
            _ => Err(()),
        }
    }
}

/// Invokes `f` on all direct fields of `ty`.
pub fn iter_fields<'tcx>(
    ty: Ty<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mut f: impl FnMut(Option<VariantIdx>, Field, Ty<'tcx>),
) {
    match ty.kind() {
        ty::Tuple(list) => {
            for (field, ty) in list.iter().enumerate() {
                f(None, field.into(), ty);
            }
        }
        ty::Adt(def, substs) => {
            if def.is_union() {
                return;
            }
            for (v_index, v_def) in def.variants().iter_enumerated() {
                let variant = if def.is_struct() { None } else { Some(v_index) };
                for (f_index, f_def) in v_def.fields.iter().enumerate() {
                    let field_ty = f_def.ty(tcx, substs);
                    let field_ty = tcx
                        .try_normalize_erasing_regions(param_env, field_ty)
                        .unwrap_or_else(|_| tcx.erase_regions(field_ty));
                    f(variant, f_index.into(), field_ty);
                }
            }
        }
        ty::Closure(_, substs) => {
            iter_fields(substs.as_closure().tupled_upvars_ty(), tcx, param_env, f);
        }
        _ => (),
    }
}

/// Returns all locals with projections that have their reference or address taken.
pub fn excluded_locals(body: &Body<'_>) -> BitSet<Local> {
    struct Collector {
        result: BitSet<Local>,
    }

    impl<'tcx> Visitor<'tcx> for Collector {
        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
            if (context.is_borrow()
                || context.is_address_of()
                || context.is_drop()
                || context == PlaceContext::MutatingUse(MutatingUseContext::AsmOutput))
                && !place.is_indirect()
            {
                // A pointer to a place could be used to access other places with the same local,
                // hence we have to exclude the local completely.
                self.result.insert(place.local);
            }
        }
    }

    let mut collector = Collector { result: BitSet::new_empty(body.local_decls.len()) };
    collector.visit_body(body);
    collector.result
}

/// This is used to visualize the dataflow analysis.
impl<'tcx, T> DebugWithContext<ValueAnalysisWrapper<T>> for State<T::Value>
where
    T: ValueAnalysis<'tcx>,
    T::Value: Debug,
{
    fn fmt_with(&self, ctxt: &ValueAnalysisWrapper<T>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            StateData::Reachable(values) => debug_with_context(values, None, ctxt.0.map(), f),
            StateData::Unreachable => write!(f, "unreachable"),
        }
    }

    fn fmt_diff_with(
        &self,
        old: &Self,
        ctxt: &ValueAnalysisWrapper<T>,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        match (&self.0, &old.0) {
            (StateData::Reachable(this), StateData::Reachable(old)) => {
                debug_with_context(this, Some(old), ctxt.0.map(), f)
            }
            _ => Ok(()), // Consider printing something here.
        }
    }
}

fn debug_with_context_rec<V: Debug + Eq>(
    place: PlaceIndex,
    place_str: &str,
    new: &IndexVec<ValueIndex, V>,
    old: Option<&IndexVec<ValueIndex, V>>,
    map: &Map,
    f: &mut Formatter<'_>,
) -> std::fmt::Result {
    if let Some(value) = map.places[place].value_index {
        match old {
            None => writeln!(f, "{}: {:?}", place_str, new[value])?,
            Some(old) => {
                if new[value] != old[value] {
                    writeln!(f, "\u{001f}-{}: {:?}", place_str, old[value])?;
                    writeln!(f, "\u{001f}+{}: {:?}", place_str, new[value])?;
                }
            }
        }
    }

    for child in map.children(place) {
        let info_elem = map.places[child].proj_elem.unwrap();
        let child_place_str = match info_elem {
            TrackElem::Discriminant => {
                format!("discriminant({})", place_str)
            }
            TrackElem::Variant(idx) => {
                format!("({} as {:?})", place_str, idx)
            }
            TrackElem::Field(field) => {
                if place_str.starts_with('*') {
                    format!("({}).{}", place_str, field.index())
                } else {
                    format!("{}.{}", place_str, field.index())
                }
            }
        };
        debug_with_context_rec(child, &child_place_str, new, old, map, f)?;
    }

    Ok(())
}

fn debug_with_context<V: Debug + Eq>(
    new: &IndexVec<ValueIndex, V>,
    old: Option<&IndexVec<ValueIndex, V>>,
    map: &Map,
    f: &mut Formatter<'_>,
) -> std::fmt::Result {
    for (local, place) in map.locals.iter_enumerated() {
        if let Some(place) = place {
            debug_with_context_rec(*place, &format!("{local:?}"), new, old, map, f)?;
        }
    }
    Ok(())
}
