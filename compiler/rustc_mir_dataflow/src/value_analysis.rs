use std::fmt::{Debug, Formatter};
use std::ops::Range;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::{FxHashMap, FxIndexSet, StdEntry};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use tracing::debug;

use crate::JoinSemiLattice;
use crate::lattice::{HasBottom, HasTop};

rustc_index::newtype_index!(
    /// This index uniquely identifies a place.
    ///
    /// Not every place has a `PlaceIndex`, and not every `PlaceIndex` corresponds to a tracked
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
pub struct StateData<V> {
    bottom: V,
    /// This map only contains values that are not `⊥`.
    map: FxHashMap<ValueIndex, V>,
}

impl<V: HasBottom> StateData<V> {
    fn new() -> StateData<V> {
        StateData { bottom: V::BOTTOM, map: FxHashMap::default() }
    }

    fn get(&self, idx: ValueIndex) -> &V {
        self.map.get(&idx).unwrap_or(&self.bottom)
    }

    fn insert(&mut self, idx: ValueIndex, elem: V) {
        if elem.is_bottom() {
            self.map.remove(&idx);
        } else {
            self.map.insert(idx, elem);
        }
    }
}

impl<V: Clone> Clone for StateData<V> {
    fn clone(&self) -> Self {
        StateData { bottom: self.bottom.clone(), map: self.map.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.map.clone_from(&source.map)
    }
}

impl<V: JoinSemiLattice + Clone> JoinSemiLattice for StateData<V> {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        #[allow(rustc::potential_query_instability)]
        for (i, v) in other.map.iter() {
            match self.map.entry(*i) {
                StdEntry::Vacant(e) => {
                    e.insert(v.clone());
                    changed = true
                }
                StdEntry::Occupied(e) => changed |= e.into_mut().join(v),
            }
        }
        changed
    }
}

/// Dataflow state.
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
pub enum State<V> {
    Unreachable,
    Reachable(StateData<V>),
}

impl<V: Clone> Clone for State<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Reachable(x) => Self::Reachable(x.clone()),
            Self::Unreachable => Self::Unreachable,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (&mut *self, source) {
            (Self::Reachable(x), Self::Reachable(y)) => {
                x.clone_from(&y);
            }
            _ => *self = source.clone(),
        }
    }
}

impl<V: Clone + HasBottom> State<V> {
    pub fn new_reachable() -> State<V> {
        State::Reachable(StateData::new())
    }

    pub fn all_bottom(&self) -> bool {
        match self {
            State::Unreachable => false,
            State::Reachable(values) =>
            {
                #[allow(rustc::potential_query_instability)]
                values.map.values().all(V::is_bottom)
            }
        }
    }

    pub fn is_reachable(&self) -> bool {
        matches!(self, State::Reachable(_))
    }

    /// Assign `value` to all places that are contained in `place` or may alias one.
    pub fn flood_with(&mut self, place: PlaceRef<'_>, map: &Map<'_>, value: V) {
        self.flood_with_tail_elem(place, None, map, value)
    }

    /// Assign `TOP` to all places that are contained in `place` or may alias one.
    pub fn flood(&mut self, place: PlaceRef<'_>, map: &Map<'_>)
    where
        V: HasTop,
    {
        self.flood_with(place, map, V::TOP)
    }

    /// Assign `value` to the discriminant of `place` and all places that may alias it.
    fn flood_discr_with(&mut self, place: PlaceRef<'_>, map: &Map<'_>, value: V) {
        self.flood_with_tail_elem(place, Some(TrackElem::Discriminant), map, value)
    }

    /// Assign `TOP` to the discriminant of `place` and all places that may alias it.
    pub fn flood_discr(&mut self, place: PlaceRef<'_>, map: &Map<'_>)
    where
        V: HasTop,
    {
        self.flood_discr_with(place, map, V::TOP)
    }

    /// This method is the most general version of the `flood_*` method.
    ///
    /// Assign `value` on the given place and all places that may alias it. In particular, when
    /// the given place has a variant downcast, we invoke the function on all the other variants.
    ///
    /// `tail_elem` allows to support discriminants that are not a place in MIR, but that we track
    /// as such.
    pub fn flood_with_tail_elem(
        &mut self,
        place: PlaceRef<'_>,
        tail_elem: Option<TrackElem>,
        map: &Map<'_>,
        value: V,
    ) {
        let State::Reachable(values) = self else { return };
        map.for_each_aliasing_place(place, tail_elem, &mut |vi| values.insert(vi, value.clone()));
    }

    /// Low-level method that assigns to a place.
    /// This does nothing if the place is not tracked.
    ///
    /// The target place must have been flooded before calling this method.
    fn insert_idx(&mut self, target: PlaceIndex, result: ValueOrPlace<V>, map: &Map<'_>) {
        match result {
            ValueOrPlace::Value(value) => self.insert_value_idx(target, value, map),
            ValueOrPlace::Place(source) => self.insert_place_idx(target, source, map),
        }
    }

    /// Low-level method that assigns a value to a place.
    /// This does nothing if the place is not tracked.
    ///
    /// The target place must have been flooded before calling this method.
    pub fn insert_value_idx(&mut self, target: PlaceIndex, value: V, map: &Map<'_>) {
        let State::Reachable(values) = self else { return };
        if let Some(value_index) = map.places[target].value_index {
            values.insert(value_index, value)
        }
    }

    /// Copies `source` to `target`, including all tracked places beneath.
    ///
    /// If `target` contains a place that is not contained in `source`, it will be overwritten with
    /// Top. Also, because this will copy all entries one after another, it may only be used for
    /// places that are non-overlapping or identical.
    ///
    /// The target place must have been flooded before calling this method.
    pub fn insert_place_idx(&mut self, target: PlaceIndex, source: PlaceIndex, map: &Map<'_>) {
        let State::Reachable(values) = self else { return };

        // If both places are tracked, we copy the value to the target.
        // If the target is tracked, but the source is not, we do nothing, as invalidation has
        // already been performed.
        if let Some(target_value) = map.places[target].value_index
            && let Some(source_value) = map.places[source].value_index
        {
            values.insert(target_value, values.get(source_value).clone());
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
    pub fn assign(&mut self, target: PlaceRef<'_>, result: ValueOrPlace<V>, map: &Map<'_>)
    where
        V: HasTop,
    {
        self.flood(target, map);
        if let Some(target) = map.find(target) {
            self.insert_idx(target, result, map);
        }
    }

    /// Helper method for assignments to a discriminant.
    pub fn assign_discr(&mut self, target: PlaceRef<'_>, result: ValueOrPlace<V>, map: &Map<'_>)
    where
        V: HasTop,
    {
        self.flood_discr(target, map);
        if let Some(target) = map.find_discr(target) {
            self.insert_idx(target, result, map);
        }
    }

    /// Retrieve the value stored for a place, or `None` if it is not tracked.
    pub fn try_get(&self, place: PlaceRef<'_>, map: &Map<'_>) -> Option<V> {
        let place = map.find(place)?;
        self.try_get_idx(place, map)
    }

    /// Retrieve the discriminant stored for a place, or `None` if it is not tracked.
    pub fn try_get_discr(&self, place: PlaceRef<'_>, map: &Map<'_>) -> Option<V> {
        let place = map.find_discr(place)?;
        self.try_get_idx(place, map)
    }

    /// Retrieve the slice length stored for a place, or `None` if it is not tracked.
    pub fn try_get_len(&self, place: PlaceRef<'_>, map: &Map<'_>) -> Option<V> {
        let place = map.find_len(place)?;
        self.try_get_idx(place, map)
    }

    /// Retrieve the value stored for a place index, or `None` if it is not tracked.
    pub fn try_get_idx(&self, place: PlaceIndex, map: &Map<'_>) -> Option<V> {
        match self {
            State::Reachable(values) => {
                map.places[place].value_index.map(|v| values.get(v).clone())
            }
            State::Unreachable => None,
        }
    }

    /// Retrieve the value stored for a place, or ⊤ if it is not tracked.
    ///
    /// This method returns ⊥ if the place is tracked and the state is unreachable.
    pub fn get(&self, place: PlaceRef<'_>, map: &Map<'_>) -> V
    where
        V: HasBottom + HasTop,
    {
        match self {
            State::Reachable(_) => self.try_get(place, map).unwrap_or(V::TOP),
            // Because this is unreachable, we can return any value we want.
            State::Unreachable => V::BOTTOM,
        }
    }

    /// Retrieve the value stored for a place, or ⊤ if it is not tracked.
    ///
    /// This method returns ⊥ the current state is unreachable.
    pub fn get_discr(&self, place: PlaceRef<'_>, map: &Map<'_>) -> V
    where
        V: HasBottom + HasTop,
    {
        match self {
            State::Reachable(_) => self.try_get_discr(place, map).unwrap_or(V::TOP),
            // Because this is unreachable, we can return any value we want.
            State::Unreachable => V::BOTTOM,
        }
    }

    /// Retrieve the value stored for a place, or ⊤ if it is not tracked.
    ///
    /// This method returns ⊥ the current state is unreachable.
    pub fn get_len(&self, place: PlaceRef<'_>, map: &Map<'_>) -> V
    where
        V: HasBottom + HasTop,
    {
        match self {
            State::Reachable(_) => self.try_get_len(place, map).unwrap_or(V::TOP),
            // Because this is unreachable, we can return any value we want.
            State::Unreachable => V::BOTTOM,
        }
    }

    /// Retrieve the value stored for a place index, or ⊤ if it is not tracked.
    ///
    /// This method returns ⊥ the current state is unreachable.
    pub fn get_idx(&self, place: PlaceIndex, map: &Map<'_>) -> V
    where
        V: HasBottom + HasTop,
    {
        match self {
            State::Reachable(values) => {
                map.places[place].value_index.map(|v| values.get(v).clone()).unwrap_or(V::TOP)
            }
            State::Unreachable => {
                // Because this is unreachable, we can return any value we want.
                V::BOTTOM
            }
        }
    }
}

impl<V: JoinSemiLattice + Clone> JoinSemiLattice for State<V> {
    fn join(&mut self, other: &Self) -> bool {
        match (&mut *self, other) {
            (_, State::Unreachable) => false,
            (State::Unreachable, _) => {
                *self = other.clone();
                true
            }
            (State::Reachable(this), State::Reachable(other)) => this.join(other),
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
pub struct Map<'tcx> {
    locals: IndexVec<Local, Option<PlaceIndex>>,
    projections: FxHashMap<(PlaceIndex, TrackElem), PlaceIndex>,
    places: IndexVec<PlaceIndex, PlaceInfo<'tcx>>,
    value_count: usize,
    // The Range corresponds to a slice into `inner_values_buffer`.
    inner_values: IndexVec<PlaceIndex, Range<usize>>,
    inner_values_buffer: Vec<ValueIndex>,
}

impl<'tcx> Map<'tcx> {
    /// Returns a map that only tracks places whose type has scalar layout.
    ///
    /// This is currently the only way to create a [`Map`]. The way in which the tracked places are
    /// chosen is an implementation detail and may not be relied upon (other than that their type
    /// are scalars).
    pub fn new(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, value_limit: Option<usize>) -> Self {
        let mut map = Self {
            locals: IndexVec::from_elem(None, &body.local_decls),
            projections: FxHashMap::default(),
            places: IndexVec::new(),
            value_count: 0,
            inner_values: IndexVec::new(),
            inner_values_buffer: Vec::new(),
        };
        let exclude = excluded_locals(body);
        map.register(tcx, body, exclude, value_limit);
        debug!("registered {} places ({} nodes in total)", map.value_count, map.places.len());
        map
    }

    /// Register all non-excluded places that have scalar layout.
    #[tracing::instrument(level = "trace", skip(self, tcx, body))]
    fn register(
        &mut self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        exclude: DenseBitSet<Local>,
        value_limit: Option<usize>,
    ) {
        // Start by constructing the places for each bare local.
        for (local, decl) in body.local_decls.iter_enumerated() {
            if exclude.contains(local) {
                continue;
            }
            if decl.ty.is_async_drop_in_place_coroutine(tcx) {
                continue;
            }

            // Create a place for the local.
            debug_assert!(self.locals[local].is_none());
            let place = self.places.push(PlaceInfo::new(decl.ty, None));
            self.locals[local] = Some(place);
        }

        // Collect syntactic places and assignments between them.
        let mut collector =
            PlaceCollector { tcx, body, map: self, assignments: Default::default() };
        collector.visit_body(body);
        let PlaceCollector { mut assignments, .. } = collector;

        // Just collecting syntactic places is not enough. We may need to propagate this pattern:
        //      _1 = (const 5u32, const 13i64);
        //      _2 = _1;
        //      _3 = (_2.0 as u32);
        //
        // `_1.0` does not appear, but we still need to track it. This is achieved by propagating
        // projections from assignments. We recorded an assignment between `_2` and `_1`, so we
        // want `_1` and `_2` to have the same sub-places.
        //
        // This is what this fixpoint loop does. While we are still creating places, run through
        // all the assignments, and register places for children.
        let mut num_places = 0;
        while num_places < self.places.len() {
            num_places = self.places.len();

            for assign in 0.. {
                let Some(&(lhs, rhs)) = assignments.get_index(assign) else { break };

                // Mirror children from `lhs` in `rhs`.
                let mut child = self.places[lhs].first_child;
                while let Some(lhs_child) = child {
                    let PlaceInfo { ty, proj_elem, next_sibling, .. } = self.places[lhs_child];
                    let rhs_child =
                        self.register_place(ty, rhs, proj_elem.expect("child is not a projection"));
                    assignments.insert((lhs_child, rhs_child));
                    child = next_sibling;
                }

                // Conversely, mirror children from `rhs` in `lhs`.
                let mut child = self.places[rhs].first_child;
                while let Some(rhs_child) = child {
                    let PlaceInfo { ty, proj_elem, next_sibling, .. } = self.places[rhs_child];
                    let lhs_child =
                        self.register_place(ty, lhs, proj_elem.expect("child is not a projection"));
                    assignments.insert((lhs_child, rhs_child));
                    child = next_sibling;
                }
            }
        }
        drop(assignments);

        // Create values for places whose type have scalar layout.
        let typing_env = body.typing_env(tcx);
        for place_info in self.places.iter_mut() {
            // The user requires a bound on the number of created values.
            if let Some(value_limit) = value_limit
                && self.value_count >= value_limit
            {
                break;
            }

            if let Ok(ty) = tcx.try_normalize_erasing_regions(typing_env, place_info.ty) {
                place_info.ty = ty;
            }

            // Allocate a value slot if it doesn't have one, and the user requested one.
            assert!(place_info.value_index.is_none());
            if let Ok(layout) = tcx.layout_of(typing_env.as_query_input(place_info.ty))
                && layout.backend_repr.is_scalar()
            {
                place_info.value_index = Some(self.value_count.into());
                self.value_count += 1;
            }
        }

        // Pre-compute the tree of ValueIndex nested in each PlaceIndex.
        // `inner_values_buffer[inner_values[place]]` is the set of all the values
        // reachable by projecting `place`.
        self.inner_values_buffer = Vec::with_capacity(self.value_count);
        self.inner_values = IndexVec::from_elem(0..0, &self.places);
        for local in body.local_decls.indices() {
            if let Some(place) = self.locals[local] {
                self.cache_preorder_invoke(place);
            }
        }

        // Trim useless places.
        for opt_place in self.locals.iter_mut() {
            if let Some(place) = *opt_place
                && self.inner_values[place].is_empty()
            {
                *opt_place = None;
            }
        }
        #[allow(rustc::potential_query_instability)]
        self.projections.retain(|_, child| !self.inner_values[*child].is_empty());
    }

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn register_place(&mut self, ty: Ty<'tcx>, base: PlaceIndex, elem: TrackElem) -> PlaceIndex {
        *self.projections.entry((base, elem)).or_insert_with(|| {
            let next = self.places.push(PlaceInfo::new(ty, Some(elem)));
            self.places[next].next_sibling = self.places[base].first_child;
            self.places[base].first_child = Some(next);
            next
        })
    }

    /// Precompute the list of values inside `root` and store it inside
    /// as a slice within `inner_values_buffer`.
    fn cache_preorder_invoke(&mut self, root: PlaceIndex) {
        let start = self.inner_values_buffer.len();
        if let Some(vi) = self.places[root].value_index {
            self.inner_values_buffer.push(vi);
        }

        // We manually iterate instead of using `children` as we need to mutate `self`.
        let mut next_child = self.places[root].first_child;
        while let Some(child) = next_child {
            ensure_sufficient_stack(|| self.cache_preorder_invoke(child));
            next_child = self.places[child].next_sibling;
        }

        let end = self.inner_values_buffer.len();
        self.inner_values[root] = start..end;
    }
}

struct PlaceCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    map: &'a mut Map<'tcx>,
    assignments: FxIndexSet<(PlaceIndex, PlaceIndex)>,
}

impl<'tcx> PlaceCollector<'_, 'tcx> {
    #[tracing::instrument(level = "trace", skip(self))]
    fn register_place(&mut self, place: Place<'tcx>) -> Option<PlaceIndex> {
        // Create a place for this projection.
        let mut place_index = self.map.locals[place.local]?;
        let mut ty = PlaceTy::from_ty(self.body.local_decls[place.local].ty);
        tracing::trace!(?place_index, ?ty);

        if let ty::Ref(_, ref_ty, _) | ty::RawPtr(ref_ty, _) = ty.ty.kind()
            && let ty::Slice(..) = ref_ty.kind()
        {
            self.map.register_place(self.tcx.types.usize, place_index, TrackElem::DerefLen);
        } else if ty.ty.is_enum() {
            let discriminant_ty = ty.ty.discriminant_ty(self.tcx);
            self.map.register_place(discriminant_ty, place_index, TrackElem::Discriminant);
        }

        for proj in place.projection {
            let track_elem = proj.try_into().ok()?;
            ty = ty.projection_ty(self.tcx, proj);
            place_index = self.map.register_place(ty.ty, place_index, track_elem);
            tracing::trace!(?proj, ?place_index, ?ty);

            if let ty::Ref(_, ref_ty, _) | ty::RawPtr(ref_ty, _) = ty.ty.kind()
                && let ty::Slice(..) = ref_ty.kind()
            {
                self.map.register_place(self.tcx.types.usize, place_index, TrackElem::DerefLen);
            } else if ty.ty.is_enum() {
                let discriminant_ty = ty.ty.discriminant_ty(self.tcx);
                self.map.register_place(discriminant_ty, place_index, TrackElem::Discriminant);
            }
        }

        Some(place_index)
    }
}

impl<'tcx> Visitor<'tcx> for PlaceCollector<'_, 'tcx> {
    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_place(&mut self, place: &Place<'tcx>, ctxt: PlaceContext, _: Location) {
        if !ctxt.is_use() {
            return;
        }

        self.register_place(*place);
    }

    fn visit_assign(&mut self, lhs: &Place<'tcx>, rhs: &Rvalue<'tcx>, location: Location) {
        self.super_assign(lhs, rhs, location);

        match rhs {
            Rvalue::Use(Operand::Move(rhs) | Operand::Copy(rhs)) | Rvalue::CopyForDeref(rhs) => {
                let Some(lhs) = self.register_place(*lhs) else { return };
                let Some(rhs) = self.register_place(*rhs) else { return };
                self.assignments.insert((lhs, rhs));
            }
            Rvalue::Aggregate(kind, fields) => {
                let Some(mut lhs) = self.register_place(*lhs) else { return };
                match **kind {
                    // Do not propagate unions.
                    AggregateKind::Adt(_, _, _, _, Some(_)) => return,
                    AggregateKind::Adt(_, variant, _, _, None) => {
                        let ty = self.map.places[lhs].ty;
                        if ty.is_enum() {
                            lhs = self.map.register_place(ty, lhs, TrackElem::Variant(variant));
                        }
                    }
                    AggregateKind::RawPtr(..)
                    | AggregateKind::Array(_)
                    | AggregateKind::Tuple
                    | AggregateKind::Closure(..)
                    | AggregateKind::Coroutine(..)
                    | AggregateKind::CoroutineClosure(..) => {}
                }
                for (index, field) in fields.iter_enumerated() {
                    if let Some(rhs) = field.place()
                        && let Some(rhs) = self.register_place(rhs)
                    {
                        let lhs = self.map.register_place(
                            self.map.places[rhs].ty,
                            lhs,
                            TrackElem::Field(index),
                        );
                        self.assignments.insert((lhs, rhs));
                    }
                }
            }
            _ => {}
        }
    }
}

impl<'tcx> Map<'tcx> {
    /// Applies a single projection element, yielding the corresponding child.
    pub fn apply(&self, place: PlaceIndex, elem: TrackElem) -> Option<PlaceIndex> {
        self.projections.get(&(place, elem)).copied()
    }

    /// Locates the given place, if it exists in the tree.
    fn find_extra(
        &self,
        place: PlaceRef<'_>,
        extra: impl IntoIterator<Item = TrackElem>,
    ) -> Option<PlaceIndex> {
        let mut index = *self.locals[place.local].as_ref()?;

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

    /// Locates the given place and applies `DerefLen`, if it exists in the tree.
    pub fn find_len(&self, place: PlaceRef<'_>) -> Option<PlaceIndex> {
        self.find_extra(place, [TrackElem::DerefLen])
    }

    /// Iterate over all direct children.
    fn children(&self, parent: PlaceIndex) -> impl Iterator<Item = PlaceIndex> {
        Children::new(self, parent)
    }

    /// Invoke a function on the given place and all places that may alias it.
    ///
    /// In particular, when the given place has a variant downcast, we invoke the function on all
    /// the other variants.
    ///
    /// `tail_elem` allows to support discriminants that are not a place in MIR, but that we track
    /// as such.
    fn for_each_aliasing_place(
        &self,
        place: PlaceRef<'_>,
        tail_elem: Option<TrackElem>,
        f: &mut impl FnMut(ValueIndex),
    ) {
        if place.is_indirect_first_projection() {
            // We do not track indirect places.
            return;
        }
        let Some(mut index) = self.locals[place.local] else {
            // The local is not tracked at all, so it does not alias anything.
            return;
        };
        let elems = place.projection.iter().map(|&elem| elem.try_into()).chain(tail_elem.map(Ok));
        for elem in elems {
            // A field aliases the parent place.
            if let Some(vi) = self.places[index].value_index {
                f(vi);
            }

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
        self.for_each_value_inside(index, f);
    }

    /// Invoke the given function on all the descendants of the given place, except one branch.
    fn for_each_variant_sibling(
        &self,
        parent: PlaceIndex,
        preserved_child: Option<PlaceIndex>,
        f: &mut impl FnMut(ValueIndex),
    ) {
        for sibling in self.children(parent) {
            let elem = self.places[sibling].proj_elem;
            // Only invalidate variants and discriminant. Fields (for coroutines) are not
            // invalidated by assignment to a variant.
            if let Some(TrackElem::Variant(..) | TrackElem::Discriminant) = elem
                // Only invalidate the other variants, the current one is fine.
                && Some(sibling) != preserved_child
            {
                self.for_each_value_inside(sibling, f);
            }
        }
    }

    /// Invoke a function on each value in the given place and all descendants.
    fn for_each_value_inside(&self, root: PlaceIndex, f: &mut impl FnMut(ValueIndex)) {
        let range = self.inner_values[root].clone();
        let values = &self.inner_values_buffer[range];
        for &v in values {
            f(v)
        }
    }

    /// Invoke a function on each value in the given place and all descendants.
    pub fn for_each_projection_value<O>(
        &self,
        root: PlaceIndex,
        value: O,
        project: &mut impl FnMut(TrackElem, &O) -> Option<O>,
        f: &mut impl FnMut(PlaceIndex, &O),
    ) {
        // Fast path is there is nothing to do.
        if self.inner_values[root].is_empty() {
            return;
        }

        if self.places[root].value_index.is_some() {
            f(root, &value)
        }

        for child in self.children(root) {
            let elem = self.places[child].proj_elem.unwrap();
            if let Some(value) = project(elem, &value) {
                self.for_each_projection_value(child, value, project, f);
            }
        }
    }
}

/// This is the information tracked for every [`PlaceIndex`] and is stored by [`Map`].
///
/// Together, `first_child` and `next_sibling` form an intrusive linked list, which is used to
/// model a tree structure (a replacement for a member like `children: Vec<PlaceIndex>`).
#[derive(Debug)]
struct PlaceInfo<'tcx> {
    /// Type of the referenced place.
    ty: Ty<'tcx>,

    /// We store a [`ValueIndex`] if and only if the placed is tracked by the analysis.
    value_index: Option<ValueIndex>,

    /// The projection used to go from parent to this node (only None for root).
    proj_elem: Option<TrackElem>,

    /// The leftmost child.
    first_child: Option<PlaceIndex>,

    /// Index of the sibling to the right of this node.
    next_sibling: Option<PlaceIndex>,
}

impl<'tcx> PlaceInfo<'tcx> {
    fn new(ty: Ty<'tcx>, proj_elem: Option<TrackElem>) -> Self {
        Self { ty, next_sibling: None, first_child: None, proj_elem, value_index: None }
    }
}

struct Children<'a, 'tcx> {
    map: &'a Map<'tcx>,
    next: Option<PlaceIndex>,
}

impl<'a, 'tcx> Children<'a, 'tcx> {
    fn new(map: &'a Map<'tcx>, parent: PlaceIndex) -> Self {
        Self { map, next: map.places[parent].first_child }
    }
}

impl Iterator for Children<'_, '_> {
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
    pub const TOP: Self = ValueOrPlace::Value(V::TOP);
}

/// The set of projection elements that can be used by a tracked place.
///
/// Although only field projections are currently allowed, this could change in the future.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackElem {
    Field(FieldIdx),
    Variant(VariantIdx),
    Discriminant,
    // Length of a slice.
    DerefLen,
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
    typing_env: ty::TypingEnv<'tcx>,
    mut f: impl FnMut(Option<VariantIdx>, FieldIdx, Ty<'tcx>),
) {
    match ty.kind() {
        ty::Tuple(list) => {
            for (field, ty) in list.iter().enumerate() {
                f(None, field.into(), ty);
            }
        }
        ty::Adt(def, args) => {
            if def.is_union() {
                return;
            }
            for (v_index, v_def) in def.variants().iter_enumerated() {
                let variant = if def.is_struct() { None } else { Some(v_index) };
                for (f_index, f_def) in v_def.fields.iter().enumerate() {
                    let field_ty = f_def.ty(tcx, args);
                    let field_ty = tcx
                        .try_normalize_erasing_regions(typing_env, field_ty)
                        .unwrap_or_else(|_| tcx.erase_and_anonymize_regions(field_ty));
                    f(variant, f_index.into(), field_ty);
                }
            }
        }
        ty::Closure(_, args) => {
            iter_fields(args.as_closure().tupled_upvars_ty(), tcx, typing_env, f);
        }
        ty::Coroutine(_, args) => {
            iter_fields(args.as_coroutine().tupled_upvars_ty(), tcx, typing_env, f);
        }
        ty::CoroutineClosure(_, args) => {
            iter_fields(args.as_coroutine_closure().tupled_upvars_ty(), tcx, typing_env, f);
        }
        _ => (),
    }
}

/// Returns all locals with projections that have their reference or address taken.
pub fn excluded_locals(body: &Body<'_>) -> DenseBitSet<Local> {
    struct Collector {
        result: DenseBitSet<Local>,
    }

    impl<'tcx> Visitor<'tcx> for Collector {
        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
            if context.may_observe_address() && !place.is_indirect() {
                // A pointer to a place could be used to access other places with the same local,
                // hence we have to exclude the local completely.
                self.result.insert(place.local);
            }
        }
    }

    let mut collector = Collector { result: DenseBitSet::new_empty(body.local_decls.len()) };
    collector.visit_body(body);
    collector.result
}

fn debug_with_context_rec<V: Debug + Eq + HasBottom>(
    place: PlaceIndex,
    place_str: &str,
    new: &StateData<V>,
    old: Option<&StateData<V>>,
    map: &Map<'_>,
    f: &mut Formatter<'_>,
) -> std::fmt::Result {
    if let Some(value) = map.places[place].value_index {
        match old {
            None => writeln!(f, "{}: {:?}", place_str, new.get(value))?,
            Some(old) => {
                if new.get(value) != old.get(value) {
                    writeln!(f, "\u{001f}-{}: {:?}", place_str, old.get(value))?;
                    writeln!(f, "\u{001f}+{}: {:?}", place_str, new.get(value))?;
                }
            }
        }
    }

    for child in map.children(place) {
        let info_elem = map.places[child].proj_elem.unwrap();
        let child_place_str = match info_elem {
            TrackElem::Discriminant => {
                format!("discriminant({place_str})")
            }
            TrackElem::Variant(idx) => {
                format!("({place_str} as {idx:?})")
            }
            TrackElem::Field(field) => {
                if place_str.starts_with('*') {
                    format!("({}).{}", place_str, field.index())
                } else {
                    format!("{}.{}", place_str, field.index())
                }
            }
            TrackElem::DerefLen => {
                format!("Len(*{})", place_str)
            }
        };
        debug_with_context_rec(child, &child_place_str, new, old, map, f)?;
    }

    Ok(())
}

pub fn debug_with_context<V: Debug + Eq + HasBottom>(
    new: &StateData<V>,
    old: Option<&StateData<V>>,
    map: &Map<'_>,
    f: &mut Formatter<'_>,
) -> std::fmt::Result {
    for (local, place) in map.locals.iter_enumerated() {
        if let Some(place) = place {
            debug_with_context_rec(*place, &format!("{local:?}"), new, old, map, f)?;
        }
    }
    Ok(())
}
