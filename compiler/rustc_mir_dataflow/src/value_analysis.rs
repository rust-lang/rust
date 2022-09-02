//! This module provides a framework on top of the normal MIR dataflow framework to simplify the
//! implementation of analyses that track the values stored in places of interest.
//!
//! The default methods of [`ValueAnalysis`] (prefixed with `super_` instead of `handle_`)
//! provide some behavior that should be valid for all abstract domains that are based only on the
//! value stored in a certain place. On top of these default rules, an implementation should
//! override some of the `handle_` methods. For an example, see `ConstAnalysis`.
//!
//! An implementation must also provide a [`Map`]. Before the anaylsis begins, all places that
//! should be tracked during the analysis must be registered. Currently, the projections of these
//! places may only contain derefs, fields and downcasts (otherwise registration fails). During the
//! analysis, no new places can be registered.
//!
//! Note that if you want to track values behind references, you have to register the dereferenced
//! place. For example: Assume `let x = (0, 0)` and that we want to propagate values from `x.0` and
//! `x.1` also through the assignment `let y = &x`. In this case, we should register `x.0`, `x.1`,
//! `(*y).0` and `(*y).1`.
//!
//!
//! # Correctness
//!
//! Warning: This is a semi-formal attempt to argue for the correctness of this analysis. If you
//! find any weak spots, let me know! Recommended reading: Abstract Interpretation. We will use the
//! term "place" to refer to a place expression (like `mir::Place`), and we will call the
//! underlying entity "object". For instance, `*_1` and `*_2` are not the same place, but depending
//! on the value of `_1` and `_2`, they could refer to the same object. Also, the same place can
//! refer to different objects during execution. If `_1` is reassigned, then `*_1` may refer to
//! different objects before and after assignment. Additionally, when saying "access to a place",
//! what we really mean is "access to an object denoted by arbitrary projections of that place".
//!
//! In the following, we will assume a constant propagation analysis. Our analysis is correct if
//! every transfer function is correct. This is the case if for every pair (f, f#) and abstract
//! state s, we have f(y(s)) <= y(f#(s)), where s is a mapping from tracked place to top, bottom or
//! a constant. Since pointers (and mutable references) are not tracked, but can be used to change
//! values in the concrete domain, f# must assume that all places that can be affected in this way
//! for a given program point are already marked with top in s (otherwise many assignments and
//! function calls would have no choice but to mark all tracked places with top). This leads us to
//! an invariant: For all possible program points where there could possibly exist means of mutable
//! access to a tracked place (in the concrete domain), this place must be assigned to top (in the
//! abstract domain). The concretization function y can be defined as expected for the constant
//! propagation analysis, although the concrete state of course contains all kinds of non-tracked
//! data. However, by the invariant above, no mutable access to tracked places that are not marked
//! with top may be introduced.
//!
//! Note that we (at least currently) do not differentiate between "this place may assume different
//! values" and "a pointer to this place escaped the analysis". However, we still want to handle
//! assignments to constants as usual for f#. This adds an assumption: Whenever we have an
//! assignment that is captured by the analysis, all mutable access to the underlying place (which
//! is not observable by the analysis) must be invalidated. This is (hopefully) covered by Stacked
//! Borrows.
//!
//! To be continued...

use std::fmt::{Debug, Formatter};

use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;

use crate::{
    fmt::DebugWithContext, lattice::FlatSet, Analysis, AnalysisDomain, CallReturnPlaces,
    JoinSemiLattice, SwitchIntEdgeEffects,
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
            }
            StatementKind::CopyNonOverlapping(..) => {
                // FIXME: What to do here?
            }
            StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Deinit(_) => {
                // Could perhaps use these.
            }
            StatementKind::Nop
            | StatementKind::Retag(..)
            | StatementKind::FakeRead(..)
            | StatementKind::Coverage(..)
            | StatementKind::AscribeUserType(..) => (),
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
                .unwrap_or(ValueOrPlaceOrRef::Unknown),
            Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                state.flood(place.as_ref(), self.map());
                ValueOrPlaceOrRef::Unknown
            }
            Rvalue::CopyForDeref(place) => {
                self.handle_operand(&Operand::Copy(*place), state).into()
            }
            _ => ValueOrPlaceOrRef::Unknown,
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
                // Do want want to handle moves different? Could flood place with bottom.
                self.map()
                    .find(place.as_ref())
                    .map(ValueOrPlace::Place)
                    .unwrap_or(ValueOrPlace::Unknown)
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

    fn handle_terminator(&self, terminator: &Terminator<'tcx>, state: &mut State<Self::Value>) {
        self.super_terminator(terminator, state)
    }

    fn super_terminator(&self, terminator: &Terminator<'tcx>, _state: &mut State<Self::Value>) {
        match &terminator.kind {
            TerminatorKind::Call { .. } | TerminatorKind::InlineAsm { .. } => {
                // Effect is applied by `handle_call_return`.
            }
            TerminatorKind::DropAndReplace { .. } | TerminatorKind::Yield { .. } => {
                // They would have an effect, but are not allowed in this phase.
                bug!("encountered disallowed terminator");
            }
            _ => {
                // The other terminators can be ignored.
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
    pub struct PlaceIndex {}
);

rustc_index::newtype_index!(
    struct ValueIndex {}
);

#[derive(PartialEq, Eq, Clone, Debug)]
enum StateData<V> {
    Reachable(IndexVec<ValueIndex, V>),
    Unreachable,
}

/// All operations on unreachable states are ignored.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct State<V>(StateData<V>);

impl<V: Clone + HasTop> State<V> {
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

    pub fn assign_place_idx(&mut self, target: PlaceIndex, source: PlaceIndex, map: &Map) {
        let StateData::Reachable(values) = &mut self.0 else { return };
        if let Some(target_value) = map.places[target].value_index {
            if let Some(source_value) = map.places[source].value_index {
                values[target_value] = values[source_value].clone();
            } else {
                values[target_value] = V::top();
            }
        }
        for target_child in map.children(target) {
            // Try to find corresponding child in source.
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
                if let Some(target_deref) = map.apply_elem(target, ProjElem::Deref) {
                    self.assign_place_idx(target_deref, source, map);
                }
            }
            ValueOrPlaceOrRef::Unknown => {
                self.flood_idx(target, map);
            }
        }
    }

    pub fn get(&self, place: PlaceRef<'_>, map: &Map) -> V {
        map.find(place).map(|place| self.get_idx(place, map)).unwrap_or(V::top())
    }

    pub fn get_idx(&self, place: PlaceIndex, map: &Map) -> V {
        match &self.0 {
            StateData::Reachable(values) => {
                map.places[place].value_index.map(|v| values[v].clone()).unwrap_or(V::top())
            }
            StateData::Unreachable => V::top(),
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

#[derive(Debug)]
pub struct Map {
    locals: IndexVec<Local, Option<PlaceIndex>>,
    projections: FxHashMap<(PlaceIndex, ProjElem), PlaceIndex>,
    places: IndexVec<PlaceIndex, PlaceInfo>,
    value_count: usize,
}

impl Map {
    pub fn new() -> Self {
        Self {
            locals: IndexVec::new(),
            projections: FxHashMap::default(),
            places: IndexVec::new(),
            value_count: 0,
        }
    }

    /// Register all places with suitable types up to a certain derefence depth (to prevent cycles).
    pub fn register_with_filter<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        source: &impl HasLocalDecls<'tcx>,
        max_derefs: u32,
        mut filter: impl FnMut(Ty<'tcx>) -> bool,
    ) {
        let mut projection = Vec::new();
        for (local, decl) in source.local_decls().iter_enumerated() {
            self.register_with_filter_rec(
                tcx,
                max_derefs,
                local,
                &mut projection,
                decl.ty,
                &mut filter,
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
    ) {
        if filter(ty) {
            // Since downcasts are currently not allowed, this might fail.
            let _ = self.register(local, projection);
        }
        if max_derefs > 0 {
            if let Some(ty::TypeAndMut { ty, .. }) = ty.builtin_deref(false) {
                projection.push(PlaceElem::Deref);
                self.register_with_filter_rec(tcx, max_derefs - 1, local, projection, ty, filter);
                projection.pop();
            }
        }
        iter_fields(ty, tcx, |variant, field, ty| {
            if let Some(variant) = variant {
                projection.push(PlaceElem::Downcast(None, variant));
            }
            projection.push(PlaceElem::Field(field, ty));
            self.register_with_filter_rec(tcx, max_derefs, local, projection, ty, filter);
            projection.pop();
            if variant.is_some() {
                projection.pop();
            }
        });
    }

    pub fn register<'tcx>(
        &mut self,
        local: Local,
        projection: &[PlaceElem<'tcx>],
    ) -> Result<(), ()> {
        // Get the base index of the local.
        let mut index =
            *self.locals.get_or_insert_with(local, || self.places.push(PlaceInfo::new(None)));

        // Apply the projection.
        for &elem in projection {
            // For now, downcast is not allowed (see #101168).
            match elem {
                PlaceElem::Downcast(..) => return Err(()),
                _ => (),
            }
            let elem = elem.try_into()?;
            index = *self.projections.entry((index, elem)).or_insert_with(|| {
                // Prepend new child to the linked list.
                let next = self.places.push(PlaceInfo::new(Some(elem)));
                self.places[next].next_sibling = self.places[index].first_child;
                self.places[index].first_child = Some(next);
                next
            });
        }

        // Allocate a value slot if it doesn't have one.
        if self.places[index].value_index.is_none() {
            self.places[index].value_index = Some(self.value_count.into());
            self.value_count += 1;
        }

        Ok(())
    }

    pub fn apply_elem(&self, place: PlaceIndex, elem: ProjElem) -> Option<PlaceIndex> {
        self.projections.get(&(place, elem)).copied()
    }

    pub fn find(&self, place: PlaceRef<'_>) -> Option<PlaceIndex> {
        let mut index = *self.locals.get(place.local)?.as_ref()?;

        for &elem in place.projection {
            index = self.apply_elem(index, elem.try_into().ok()?)?;
        }

        Some(index)
    }

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

#[derive(Debug)]
struct PlaceInfo {
    next_sibling: Option<PlaceIndex>,
    first_child: Option<PlaceIndex>,
    /// The projection used to go from parent to this node (only None for root).
    proj_elem: Option<ProjElem>,
    value_index: Option<ValueIndex>,
}

impl PlaceInfo {
    fn new(proj_elem: Option<ProjElem>) -> Self {
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

// FIXME: See if we can get rid of `Unknown`.
pub enum ValueOrPlace<V> {
    Value(V),
    Place(PlaceIndex),
    Unknown,
}

pub enum ValueOrPlaceOrRef<V> {
    Value(V),
    Place(PlaceIndex),
    Ref(PlaceIndex),
    Unknown,
}

impl<V> From<ValueOrPlace<V>> for ValueOrPlaceOrRef<V> {
    fn from(x: ValueOrPlace<V>) -> Self {
        match x {
            ValueOrPlace::Value(value) => ValueOrPlaceOrRef::Value(value),
            ValueOrPlace::Place(place) => ValueOrPlaceOrRef::Place(place),
            ValueOrPlace::Unknown => ValueOrPlaceOrRef::Unknown,
        }
    }
}

pub trait HasBottom {
    fn bottom() -> Self;
}

pub trait HasTop {
    fn top() -> Self;
}

impl<V> HasBottom for FlatSet<V> {
    fn bottom() -> Self {
        Self::Bottom
    }
}

impl<V> HasTop for FlatSet<V> {
    fn top() -> Self {
        Self::Top
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ProjElem {
    Deref,
    Field(Field),
    Downcast(VariantIdx),
}

impl<V, T> TryFrom<ProjectionElem<V, T>> for ProjElem {
    type Error = ();

    fn try_from(value: ProjectionElem<V, T>) -> Result<Self, Self::Error> {
        match value {
            ProjectionElem::Deref => Ok(ProjElem::Deref),
            ProjectionElem::Field(field, _) => Ok(ProjElem::Field(field)),
            ProjectionElem::Downcast(_, variant) => Ok(ProjElem::Downcast(variant)),
            _ => Err(()),
        }
    }
}

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
            ProjElem::Deref => format!("*{}", place_str),
            ProjElem::Field(field) => {
                if place_str.starts_with("*") {
                    format!("({}).{}", place_str, field.index())
                } else {
                    format!("{}.{}", place_str, field.index())
                }
            }
            ProjElem::Downcast(variant) => format!("({} as #{})", place_str, variant.index()),
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
