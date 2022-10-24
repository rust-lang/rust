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
//! This framework is currently experimental. In particular, the features related to references are
//! currently guarded behind `-Zunsound-mir-opts`, because their correctness relies on Stacked
//! Borrows. Also, only places with scalar types can be tracked currently. This is because scalar
//! types are indivisible, which simplifies the current implementation. But this limitation could be
//! lifted in the future.
//!
//!
//! # Notes
//!
//! - The bottom state denotes uninitialized memory. Because we are only doing a sound approximation
//! of the actual execution, we can also use this state for places where access would be UB.
//!
//! - The assignment logic in `State::assign_place_idx` assumes that the places are non-overlapping,
//! or identical. Note that this refers to place expressions, not memory locations.
//!
//! - Since pointers (and mutable references) are not tracked, but can be used to change the
//! underlying values, we are conservative and immediately flood the referenced place upon creation
//! of the pointer. Also, we have to uphold the invariant that the place must stay that way as long
//! as this mutable access could exist. However...
//!
//! - Without an aliasing model like Stacked Borrows (i.e., `-Zunsound-mir-opts` is not given),
//! such mutable access is never revoked. And even shared references could be used to obtain the
//! address of a value an modify it. When not assuming Stacked Borrows, we prevent such places from
//! being tracked at all. This means that the analysis itself can assume that writes to a *tracked*
//! place always invalidate all other means of mutable access, regardless of the aliasing model.
//!
//! - Likewise, the analysis itself assumes that if the value of a *tracked* place behind a shared
//! reference is changed, the reference may not be used to access that value anymore. This is true
//! for all places if the referenced type is `Freeze` and we assume Stacked Borrows. If we are not
//! assuming Stacking Borrows (or if the referenced type could be `!Freeze`), we again prevent such
//! places from being tracked at all, making this assertion trivially true.

use std::fmt::{Debug, Formatter};

use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
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
            StatementKind::SetDiscriminant { .. } => {
                // Could treat this as writing a constant to a pseudo-place.
                // But discriminants are currently not tracked, so we do nothing.
                // Related: https://github.com/rust-lang/unsafe-code-guidelines/issues/84
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
                // A retag modifies the provenance of references. Currently references are only
                // tracked if `-Zunsound-mir-opts` is given, but this might change in the future.
                // However, it is still unclear how retags should be handled:
                // https://github.com/rust-lang/rust/pull/101168#discussion_r985304895
            }
            StatementKind::Nop
            | StatementKind::FakeRead(..)
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
    ) -> ValueOrPlaceOrRef<Self::Value> {
        self.super_rvalue(rvalue, state)
    }

    fn super_rvalue(
        &self,
        rvalue: &Rvalue<'tcx>,
        state: &mut State<Self::Value>,
    ) -> ValueOrPlaceOrRef<Self::Value> {
        match rvalue {
            Rvalue::Use(operand) => self.handle_operand(operand, state).into(),
            Rvalue::Ref(_, BorrowKind::Shared, place) => self
                .map()
                .find(place.as_ref())
                .map(ValueOrPlaceOrRef::Ref)
                .unwrap_or(ValueOrPlaceOrRef::top()),
            Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                // This is not a `&x` reference and could be used for modification.
                state.flood(place.as_ref(), self.map());
                ValueOrPlaceOrRef::top()
            }
            Rvalue::CopyForDeref(place) => {
                self.handle_operand(&Operand::Copy(*place), state).into()
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
                ValueOrPlaceOrRef::top()
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
                // Place can still be accessed after drop, and drop has mutable access to it.
                state.flood(place.as_ref(), self.map());
            }
            TerminatorKind::DropAndReplace { .. } | TerminatorKind::Yield { .. } => {
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
#[derive(PartialEq, Eq, Clone, Debug)]
enum StateData<V> {
    Reachable(IndexVec<ValueIndex, V>),
    Unreachable,
}

/// The dataflow state for an instance of [`ValueAnalysis`].
///
/// Every instance specifies a lattice that represents the possible values of a single tracked
/// place. If we call this lattice `V` and set set of tracked places `P`, then a [`State`] is an
/// element of `{unreachable} ∪ (P -> V)`. This again forms a lattice, where the bottom element is
/// `unreachable` and the top element is the mapping `p ↦ ⊤`. Note that the mapping `p ↦ ⊥` is not
/// the bottom element (because joining an unreachable and any other reachable state yields a
/// reachable state). All operations on unreachable states are ignored.
///
/// Flooding means assigning a value (by default `⊤`) to all tracked projections of a given place.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct State<V>(StateData<V>);

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
        if let Some(root) = map.find(place) {
            self.flood_idx_with(root, map, value);
        }
    }

    pub fn flood(&mut self, place: PlaceRef<'_>, map: &Map) {
        self.flood_with(place, map, V::top())
    }

    pub fn flood_idx_with(&mut self, place: PlaceIndex, map: &Map, value: V) {
        let StateData::Reachable(values) = &mut self.0 else { return };
        map.preorder_invoke(place, &mut |place| {
            if let Some(vi) = map.places[place].value_index {
                values[vi] = value.clone();
            }
        });
    }

    pub fn flood_idx(&mut self, place: PlaceIndex, map: &Map) {
        self.flood_idx_with(place, map, V::top())
    }

    /// Copies `source` to `target`, including all tracked places beneath.
    ///
    /// If `target` contains a place that is not contained in `source`, it will be overwritten with
    /// Top. Also, because this will copy all entries one after another, it may only be used for
    /// places that are non-overlapping or identical.
    pub fn assign_place_idx(&mut self, target: PlaceIndex, source: PlaceIndex, map: &Map) {
        let StateData::Reachable(values) = &mut self.0 else { return };

        // If both places are tracked, we copy the value to the target. If the target is tracked,
        // but the source is not, we have to invalidate the value in target. If the target is not
        // tracked, then we don't have to do anything.
        if let Some(target_value) = map.places[target].value_index {
            if let Some(source_value) = map.places[source].value_index {
                values[target_value] = values[source_value].clone();
            } else {
                values[target_value] = V::top();
            }
        }
        for target_child in map.children(target) {
            // Try to find corresponding child and recurse. Reasoning is similar as above.
            let projection = map.places[target_child].proj_elem.unwrap();
            if let Some(source_child) = map.projections.get(&(source, projection)) {
                self.assign_place_idx(target_child, *source_child, map);
            } else {
                self.flood_idx(target_child, map);
            }
        }
    }

    pub fn assign(&mut self, target: PlaceRef<'_>, result: ValueOrPlaceOrRef<V>, map: &Map) {
        if let Some(target) = map.find(target) {
            self.assign_idx(target, result, map);
        } else {
            // We don't track this place nor any projections, assignment can be ignored.
        }
    }

    pub fn assign_idx(&mut self, target: PlaceIndex, result: ValueOrPlaceOrRef<V>, map: &Map) {
        match result {
            ValueOrPlaceOrRef::Value(value) => {
                // First flood the target place in case we also track any projections (although
                // this scenario is currently not well-supported by the API).
                self.flood_idx(target, map);
                let StateData::Reachable(values) = &mut self.0 else { return };
                if let Some(value_index) = map.places[target].value_index {
                    values[value_index] = value;
                }
            }
            ValueOrPlaceOrRef::Place(source) => self.assign_place_idx(target, source, map),
            ValueOrPlaceOrRef::Ref(source) => {
                let StateData::Reachable(values) = &mut self.0 else { return };
                if let Some(value_index) = map.places[target].value_index {
                    values[value_index] = V::top();
                }
                // Instead of tracking of *where* a reference points to (as in, which memory
                // location), we track *what* it points to (as in, what do we know about the
                // target). For an assignment `x = &y`, we thus copy the info of `y` to `*x`.
                if let Some(target_deref) = map.apply(target, TrackElem::Deref) {
                    // We know here that `*x` is `Freeze`, because we only track through
                    // dereferences if the target type is `Freeze`.
                    self.assign_place_idx(target_deref, source, map);
                }
            }
        }
    }

    /// Retrieve the value stored for a place, or ⊥ if it is not tracked.
    pub fn get(&self, place: PlaceRef<'_>, map: &Map) -> V {
        map.find(place).map(|place| self.get_idx(place, map)).unwrap_or(V::top())
    }

    /// Retrieve the value stored for a place index, or ⊥ if it is not tracked.
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

/// A partial mapping from `Place` to `PlaceIndex`, where some place indices have value indices.
///
/// Some additional bookkeeping is done to speed up traversal:
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
    /// chosen is an implementation detail and may not be relied upon.
    pub fn from_filter<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        filter: impl FnMut(Ty<'tcx>) -> bool,
    ) -> Self {
        let mut map = Self::new();

        // If `-Zunsound-mir-opts` is given, tracking through references, and tracking of places
        // that have their reference taken is allowed. This would be "unsound" in the sense that
        // the correctness relies on an aliasing model similar to Stacked Borrows (which is
        // not yet guaranteed).
        if tcx.sess.opts.unstable_opts.unsound_mir_opts {
            // We might want to add additional limitations. If a struct has 10 boxed fields of
            // itself, there will currently be `10.pow(max_derefs)` tracked places.
            map.register_with_filter(tcx, body, 2, filter, &[]);
        } else {
            map.register_with_filter(tcx, body, 0, filter, &escaped_places(body));
        }

        map
    }

    /// Register all non-excluded places that pass the filter, up to a certain dereference depth.
    fn register_with_filter<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        max_derefs: u32,
        mut filter: impl FnMut(Ty<'tcx>) -> bool,
        exclude: &[Place<'tcx>],
    ) {
        // This is used to tell whether a type is `!Freeze`.
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());

        let mut projection = Vec::new();
        for (local, decl) in body.local_decls.iter_enumerated() {
            self.register_with_filter_rec(
                tcx,
                max_derefs,
                local,
                &mut projection,
                decl.ty,
                &mut filter,
                param_env,
                exclude,
            );
        }
    }

    fn register_with_filter_rec<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        max_derefs: u32,
        local: Local,
        projection: &mut Vec<PlaceElem<'tcx>>,
        ty: Ty<'tcx>,
        filter: &mut impl FnMut(Ty<'tcx>) -> bool,
        param_env: ty::ParamEnv<'tcx>,
        exclude: &[Place<'tcx>],
    ) {
        // This currently does a linear scan, could be improved.
        if exclude.contains(&Place { local, projection: tcx.intern_place_elems(projection) }) {
            return;
        }

        if filter(ty) {
            // This might fail if `ty` is not scalar.
            let _ = self.register_with_ty(local, projection, ty);
        }

        if max_derefs > 0 {
            if let Some(ty::TypeAndMut { ty: deref_ty, .. }) = ty.builtin_deref(false) {
                // Values behind references can only be tracked if the target is `Freeze`.
                if deref_ty.is_freeze(tcx.at(DUMMY_SP), param_env) {
                    projection.push(PlaceElem::Deref);
                    self.register_with_filter_rec(
                        tcx,
                        max_derefs - 1,
                        local,
                        projection,
                        deref_ty,
                        filter,
                        param_env,
                        exclude,
                    );
                    projection.pop();
                }
            }
        }
        iter_fields(ty, tcx, |variant, field, ty| {
            if variant.is_some() {
                // Downcasts are currently not supported.
                return;
            }
            projection.push(PlaceElem::Field(field, ty));
            self.register_with_filter_rec(
                tcx, max_derefs, local, projection, ty, filter, param_env, exclude,
            );
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

    #[allow(unused)]
    fn register<'tcx>(
        &mut self,
        local: Local,
        projection: &[PlaceElem<'tcx>],
        decls: &impl HasLocalDecls<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> Result<(), ()> {
        projection
            .iter()
            .fold(PlaceTy::from_ty(decls.local_decls()[local].ty), |place_ty, &elem| {
                place_ty.projection_ty(tcx, elem)
            });

        let place_ty = Place::ty_from(local, projection, decls, tcx);
        if place_ty.variant_index.is_some() {
            return Err(());
        }
        self.register_with_ty(local, projection, place_ty.ty)
    }

    /// Tries to track the given place. Fails if type is non-scalar or projection is not trackable.
    fn register_with_ty<'tcx>(
        &mut self,
        local: Local,
        projection: &[PlaceElem<'tcx>],
        ty: Ty<'tcx>,
    ) -> Result<(), ()> {
        if !ty.is_scalar() {
            // Currently, only scalar types are allowed, because they are atomic
            // and therefore do not require invalidation of parent places.
            return Err(());
        }

        let place = self.make_place(local, projection)?;

        // Allocate a value slot if it doesn't have one.
        if self.places[place].value_index.is_none() {
            self.places[place].value_index = Some(self.value_count.into());
            self.value_count += 1;
        }

        Ok(())
    }

    pub fn apply(&self, place: PlaceIndex, elem: TrackElem) -> Option<PlaceIndex> {
        self.projections.get(&(place, elem)).copied()
    }

    pub fn find(&self, place: PlaceRef<'_>) -> Option<PlaceIndex> {
        let mut index = *self.locals.get(place.local)?.as_ref()?;

        for &elem in place.projection {
            index = self.apply(index, elem.try_into().ok()?)?;
        }

        Some(index)
    }

    /// Iterate over all direct children.
    pub fn children(&self, parent: PlaceIndex) -> impl Iterator<Item = PlaceIndex> + '_ {
        Children::new(self, parent)
    }

    pub fn preorder_invoke(&self, root: PlaceIndex, f: &mut impl FnMut(PlaceIndex)) {
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

/// Used as the result of an operand.
pub enum ValueOrPlace<V> {
    Value(V),
    Place(PlaceIndex),
}

impl<V: HasTop> ValueOrPlace<V> {
    pub fn top() -> Self {
        ValueOrPlace::Value(V::top())
    }
}

/// Used as the result of an r-value.
pub enum ValueOrPlaceOrRef<V> {
    Value(V),
    Place(PlaceIndex),
    Ref(PlaceIndex),
}

impl<V: HasTop> ValueOrPlaceOrRef<V> {
    pub fn top() -> Self {
        ValueOrPlaceOrRef::Value(V::top())
    }
}

impl<V> From<ValueOrPlace<V>> for ValueOrPlaceOrRef<V> {
    fn from(x: ValueOrPlace<V>) -> Self {
        match x {
            ValueOrPlace::Value(value) => ValueOrPlaceOrRef::Value(value),
            ValueOrPlace::Place(place) => ValueOrPlaceOrRef::Place(place),
        }
    }
}

/// The set of projection elements that can be used by a tracked place.
///
/// For now, downcast is not allowed due to aliasing between variants (see #101168).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackElem {
    Deref,
    Field(Field),
}

impl<V, T> TryFrom<ProjectionElem<V, T>> for TrackElem {
    type Error = ();

    fn try_from(value: ProjectionElem<V, T>) -> Result<Self, Self::Error> {
        match value {
            ProjectionElem::Deref => Ok(TrackElem::Deref),
            ProjectionElem::Field(field, _) => Ok(TrackElem::Field(field)),
            _ => Err(()),
        }
    }
}

/// Invokes `f` on all direct fields of `ty`.
fn iter_fields<'tcx>(
    ty: Ty<'tcx>,
    tcx: TyCtxt<'tcx>,
    mut f: impl FnMut(Option<VariantIdx>, Field, Ty<'tcx>),
) {
    match ty.kind() {
        ty::Tuple(list) => {
            for (field, ty) in list.iter().enumerate() {
                f(None, field.into(), ty);
            }
        }
        ty::Adt(def, substs) => {
            for (v_index, v_def) in def.variants().iter_enumerated() {
                for (f_index, f_def) in v_def.fields.iter().enumerate() {
                    let field_ty = f_def.ty(tcx, substs);
                    let field_ty = tcx
                        .try_normalize_erasing_regions(ty::ParamEnv::reveal_all(), field_ty)
                        .unwrap_or(field_ty);
                    f(Some(v_index), f_index.into(), field_ty);
                }
            }
        }
        ty::Closure(_, substs) => {
            iter_fields(substs.as_closure().tupled_upvars_ty(), tcx, f);
        }
        _ => (),
    }
}

/// Returns all places, that have their reference or address taken.
///
/// This includes shared references.
fn escaped_places<'tcx>(body: &Body<'tcx>) -> Vec<Place<'tcx>> {
    struct Collector<'tcx> {
        result: Vec<Place<'tcx>>,
    }

    impl<'tcx> Visitor<'tcx> for Collector<'tcx> {
        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
            if context.is_borrow() || context.is_address_of() {
                self.result.push(*place);
            }
        }
    }

    let mut collector = Collector { result: Vec::new() };
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
            TrackElem::Deref => format!("*{}", place_str),
            TrackElem::Field(field) => {
                if place_str.starts_with("*") {
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
            debug_with_context_rec(*place, &format!("{:?}", local), new, old, map, f)?;
        }
    }
    Ok(())
}
