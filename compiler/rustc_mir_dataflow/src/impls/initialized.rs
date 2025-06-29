use std::assert_matches::assert_matches;

use rustc_abi::VariantIdx;
use rustc_index::Idx;
use rustc_index::bit_set::{DenseBitSet, MixedBitSet};
use rustc_middle::bug;
use rustc_middle::mir::{
    self, Body, CallReturnPlaces, Location, SwitchTargetValue, TerminatorEdges,
};
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{self, TyCtxt};
use smallvec::SmallVec;
use tracing::{debug, instrument};

use crate::drop_flag_effects::{DropFlagState, InactiveVariants};
use crate::move_paths::{HasMoveData, InitIndex, InitKind, LookupResult, MoveData, MovePathIndex};
use crate::{
    Analysis, GenKill, MaybeReachable, drop_flag_effects, drop_flag_effects_for_function_entry,
    drop_flag_effects_for_location, on_all_children_bits, on_lookup_result_bits,
};

// Used by both `MaybeInitializedPlaces` and `MaybeUninitializedPlaces`.
pub struct MaybePlacesSwitchIntData<'tcx> {
    enum_place: mir::Place<'tcx>,
    discriminants: Vec<(VariantIdx, Discr<'tcx>)>,
    index: usize,
}

impl<'tcx> MaybePlacesSwitchIntData<'tcx> {
    /// Creates a `SmallVec` mapping each target in `targets` to its `VariantIdx`.
    fn variants(&mut self, targets: &mir::SwitchTargets) -> SmallVec<[VariantIdx; 4]> {
        self.index = 0;
        targets.all_values().iter().map(|value| self.next_discr(value.get())).collect()
    }

    // The discriminant order in the `SwitchInt` targets should match the order yielded by
    // `AdtDef::discriminants`. We rely on this to match each discriminant in the targets to its
    // corresponding variant in linear time.
    fn next_discr(&mut self, value: u128) -> VariantIdx {
        // An out-of-bounds abort will occur if the discriminant ordering isn't as described above.
        loop {
            let (variant, discr) = self.discriminants[self.index];
            self.index += 1;
            if discr.val == value {
                return variant;
            }
        }
    }
}

impl<'tcx> MaybePlacesSwitchIntData<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        block: mir::BasicBlock,
        discr: &mir::Operand<'tcx>,
    ) -> Option<Self> {
        let Some(discr) = discr.place() else { return None };

        // Inspect a `SwitchInt`-terminated basic block to see if the condition of that `SwitchInt`
        // is an enum discriminant.
        //
        // We expect such blocks to have a call to `discriminant` as their last statement like so:
        // ```text
        // ...
        // _42 = discriminant(_1)
        // SwitchInt(_42, ..)
        // ```
        // If the basic block matches this pattern, this function gathers the place corresponding
        // to the enum (`_1` in the example above) as well as the discriminants.
        let block_data = &body[block];
        for statement in block_data.statements.iter().rev() {
            match statement.kind {
                mir::StatementKind::Assign(box (lhs, mir::Rvalue::Discriminant(enum_place)))
                    if lhs == discr =>
                {
                    match enum_place.ty(body, tcx).ty.kind() {
                        ty::Adt(enum_def, _) => {
                            return Some(MaybePlacesSwitchIntData {
                                enum_place,
                                discriminants: enum_def.discriminants(tcx).collect(),
                                index: 0,
                            });
                        }

                        // `Rvalue::Discriminant` is also used to get the active yield point for a
                        // coroutine, but we do not need edge-specific effects in that case. This
                        // may change in the future.
                        ty::Coroutine(..) => break,

                        t => bug!("`discriminant` called on unexpected type {:?}", t),
                    }
                }
                mir::StatementKind::Coverage(_) => continue,
                _ => break,
            }
        }
        None
    }
}

/// `MaybeInitializedPlaces` tracks all places that might be
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// #[rustfmt::skip]
/// fn foo(pred: bool) {                        // maybe-init:
///                                             // {}
///     let a = S; let mut b = S; let c; let d; // {a, b}
///
///     if pred {
///         drop(a);                            // {   b}
///         b = S;                              // {   b}
///
///     } else {
///         drop(b);                            // {a}
///         d = S;                              // {a,       d}
///
///     }                                       // {a, b,    d}
///
///     c = S;                                  // {a, b, c, d}
/// }
/// ```
///
/// To determine whether a place is *definitely* initialized at a
/// particular control-flow point, one can take the set-complement
/// of the data from `MaybeUninitializedPlaces` at the corresponding
/// control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeUninitializedPlaces` yields the set of
/// places that would require a dynamic drop-flag at that statement.
pub struct MaybeInitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,
    exclude_inactive_in_otherwise: bool,
    skip_unreachable_unwind: bool,
}

impl<'a, 'tcx> MaybeInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, move_data: &'a MoveData<'tcx>) -> Self {
        MaybeInitializedPlaces {
            tcx,
            body,
            move_data,
            exclude_inactive_in_otherwise: false,
            skip_unreachable_unwind: false,
        }
    }

    /// Ensures definitely inactive variants are excluded from the set of initialized places for
    /// blocks reached through an `otherwise` edge.
    pub fn exclude_inactive_in_otherwise(mut self) -> Self {
        self.exclude_inactive_in_otherwise = true;
        self
    }

    pub fn skipping_unreachable_unwind(mut self) -> Self {
        self.skip_unreachable_unwind = true;
        self
    }

    pub fn is_unwind_dead(
        &self,
        place: mir::Place<'tcx>,
        state: &<Self as Analysis<'tcx>>::Domain,
    ) -> bool {
        if let LookupResult::Exact(path) = self.move_data().rev_lookup.find(place.as_ref()) {
            let mut maybe_live = false;
            on_all_children_bits(self.move_data(), path, |child| {
                maybe_live |= state.contains(child);
            });
            !maybe_live
        } else {
            false
        }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for MaybeInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        self.move_data
    }
}

/// `MaybeUninitializedPlaces` tracks all places that might be
/// uninitialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// #[rustfmt::skip]
/// fn foo(pred: bool) {                        // maybe-uninit:
///                                             // {a, b, c, d}
///     let a = S; let mut b = S; let c; let d; // {      c, d}
///
///     if pred {
///         drop(a);                            // {a,    c, d}
///         b = S;                              // {a,    c, d}
///
///     } else {
///         drop(b);                            // {   b, c, d}
///         d = S;                              // {   b, c   }
///
///     }                                       // {a, b, c, d}
///
///     c = S;                                  // {a, b,    d}
/// }
/// ```
///
/// To determine whether a place is *definitely* uninitialized at a
/// particular control-flow point, one can take the set-complement
/// of the data from `MaybeInitializedPlaces` at the corresponding
/// control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeInitializedPlaces` yields the set of
/// places that would require a dynamic drop-flag at that statement.
pub struct MaybeUninitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,

    mark_inactive_variants_as_uninit: bool,
    include_inactive_in_otherwise: bool,
    skip_unreachable_unwind: DenseBitSet<mir::BasicBlock>,
}

impl<'a, 'tcx> MaybeUninitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, move_data: &'a MoveData<'tcx>) -> Self {
        MaybeUninitializedPlaces {
            tcx,
            body,
            move_data,
            mark_inactive_variants_as_uninit: false,
            include_inactive_in_otherwise: false,
            skip_unreachable_unwind: DenseBitSet::new_empty(body.basic_blocks.len()),
        }
    }

    /// Causes inactive enum variants to be marked as "maybe uninitialized" after a switch on an
    /// enum discriminant.
    ///
    /// This is correct in a vacuum but is not the default because it causes problems in the borrow
    /// checker, where this information gets propagated along `FakeEdge`s.
    pub fn mark_inactive_variants_as_uninit(mut self) -> Self {
        self.mark_inactive_variants_as_uninit = true;
        self
    }

    /// Ensures definitely inactive variants are included in the set of uninitialized places for
    /// blocks reached through an `otherwise` edge.
    pub fn include_inactive_in_otherwise(mut self) -> Self {
        self.include_inactive_in_otherwise = true;
        self
    }

    pub fn skipping_unreachable_unwind(
        mut self,
        unreachable_unwind: DenseBitSet<mir::BasicBlock>,
    ) -> Self {
        self.skip_unreachable_unwind = unreachable_unwind;
        self
    }
}

impl<'tcx> HasMoveData<'tcx> for MaybeUninitializedPlaces<'_, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        self.move_data
    }
}

/// `EverInitializedPlaces` tracks all places that might have ever been
/// initialized upon reaching a particular point in the control flow
/// for a function, without an intervening `StorageDead`.
///
/// This dataflow is used to determine if an immutable local variable may
/// be assigned to.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// #[rustfmt::skip]
/// fn foo(pred: bool) {                        // ever-init:
///                                             // {          }
///     let a = S; let mut b = S; let c; let d; // {a, b      }
///
///     if pred {
///         drop(a);                            // {a, b,     }
///         b = S;                              // {a, b,     }
///
///     } else {
///         drop(b);                            // {a, b,      }
///         d = S;                              // {a, b,    d }
///
///     }                                       // {a, b,    d }
///
///     c = S;                                  // {a, b, c, d }
/// }
/// ```
pub struct EverInitializedPlaces<'a, 'tcx> {
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,
}

impl<'a, 'tcx> EverInitializedPlaces<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>, move_data: &'a MoveData<'tcx>) -> Self {
        EverInitializedPlaces { body, move_data }
    }
}

impl<'tcx> HasMoveData<'tcx> for EverInitializedPlaces<'_, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        self.move_data
    }
}

impl<'a, 'tcx> MaybeInitializedPlaces<'a, 'tcx> {
    fn update_bits(
        state: &mut <Self as Analysis<'tcx>>::Domain,
        path: MovePathIndex,
        dfstate: DropFlagState,
    ) {
        match dfstate {
            DropFlagState::Absent => state.kill(path),
            DropFlagState::Present => state.gen_(path),
        }
    }
}

impl<'tcx> MaybeUninitializedPlaces<'_, 'tcx> {
    fn update_bits(
        state: &mut <Self as Analysis<'tcx>>::Domain,
        path: MovePathIndex,
        dfstate: DropFlagState,
    ) {
        match dfstate {
            DropFlagState::Absent => state.gen_(path),
            DropFlagState::Present => state.kill(path),
        }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeInitializedPlaces<'_, 'tcx> {
    /// There can be many more `MovePathIndex` than there are locals in a MIR body.
    /// We use a mixed bitset to avoid paying too high a memory footprint.
    type Domain = MaybeReachable<MixedBitSet<MovePathIndex>>;

    type SwitchIntData = MaybePlacesSwitchIntData<'tcx>;

    const NAME: &'static str = "maybe_init";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = uninitialized
        MaybeReachable::Unreachable
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut Self::Domain) {
        *state =
            MaybeReachable::Reachable(MixedBitSet::new_empty(self.move_data().move_paths.len()));
        drop_flag_effects_for_function_entry(self.body, self.move_data, |path, s| {
            assert!(s == DropFlagState::Present);
            state.gen_(path);
        });
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.body, self.move_data, location, |path, s| {
            Self::update_bits(state, path, s)
        });

        // Mark all places as "maybe init" if they are mutably borrowed. See #90752.
        if self.tcx.sess.opts.unstable_opts.precise_enum_drop_elaboration
            && let Some((_, rvalue)) = statement.kind.as_assign()
            && let mir::Rvalue::Ref(_, mir::BorrowKind::Mut { .. }, place)
                // FIXME: Does `&raw const foo` allow mutation? See #90413.
                | mir::Rvalue::RawPtr(_, place) = rvalue
            && let LookupResult::Exact(mpi) = self.move_data().rev_lookup.find(place.as_ref())
        {
            on_all_children_bits(self.move_data(), mpi, |child| {
                state.gen_(child);
            })
        }
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        // Note: `edges` must be computed first because `drop_flag_effects_for_location` can change
        // the result of `is_unwind_dead`.
        let mut edges = terminator.edges();
        if self.skip_unreachable_unwind
            && let mir::TerminatorKind::Drop {
                target,
                unwind,
                place,
                replace: _,
                drop: _,
                async_fut: _,
            } = terminator.kind
            && matches!(unwind, mir::UnwindAction::Cleanup(_))
            && self.is_unwind_dead(place, state)
        {
            edges = TerminatorEdges::Single(target);
        }
        drop_flag_effects_for_location(self.body, self.move_data, location, |path, s| {
            Self::update_bits(state, path, s)
        });
        edges
    }

    fn apply_call_return_effect(
        &mut self,
        state: &mut Self::Domain,
        _block: mir::BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            // when a call returns successfully, that means we need to set
            // the bits for that dest_place to 1 (initialized).
            on_lookup_result_bits(
                self.move_data(),
                self.move_data().rev_lookup.find(place.as_ref()),
                |mpi| {
                    state.gen_(mpi);
                },
            );
        });
    }

    fn get_switch_int_data(
        &mut self,
        block: mir::BasicBlock,
        discr: &mir::Operand<'tcx>,
    ) -> Option<Self::SwitchIntData> {
        if !self.tcx.sess.opts.unstable_opts.precise_enum_drop_elaboration {
            return None;
        }

        MaybePlacesSwitchIntData::new(self.tcx, self.body, block, discr)
    }

    fn apply_switch_int_edge_effect(
        &mut self,
        data: &mut Self::SwitchIntData,
        state: &mut Self::Domain,
        value: SwitchTargetValue,
        targets: &mir::SwitchTargets,
    ) {
        let inactive_variants = match value {
            SwitchTargetValue::Normal(value) => InactiveVariants::Active(data.next_discr(value)),
            SwitchTargetValue::Otherwise if self.exclude_inactive_in_otherwise => {
                InactiveVariants::Inactives(data.variants(targets))
            }
            _ => return,
        };

        // Kill all move paths that correspond to variants we know to be inactive along this
        // particular outgoing edge of a `SwitchInt`.
        drop_flag_effects::on_all_inactive_variants(
            self.move_data,
            data.enum_place,
            &inactive_variants,
            |mpi| state.kill(mpi),
        );
    }
}

/// There can be many more `MovePathIndex` than there are locals in a MIR body.
/// We use a mixed bitset to avoid paying too high a memory footprint.
pub type MaybeUninitializedPlacesDomain = MixedBitSet<MovePathIndex>;

impl<'tcx> Analysis<'tcx> for MaybeUninitializedPlaces<'_, 'tcx> {
    type Domain = MaybeUninitializedPlacesDomain;

    type SwitchIntData = MaybePlacesSwitchIntData<'tcx>;

    const NAME: &'static str = "maybe_uninit";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = initialized (`initialize_start_block` overwrites this on first entry)
        MixedBitSet::new_empty(self.move_data().move_paths.len())
    }

    // sets state bits for Arg places
    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut Self::Domain) {
        // set all bits to 1 (uninit) before gathering counter-evidence
        state.insert_all();

        drop_flag_effects_for_function_entry(self.body, self.move_data, |path, s| {
            assert!(s == DropFlagState::Present);
            state.remove(path);
        });
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.body, self.move_data, location, |path, s| {
            Self::update_bits(state, path, s)
        });

        // Unlike in `MaybeInitializedPlaces` above, we don't need to change the state when a
        // mutable borrow occurs. Places cannot become uninitialized through a mutable reference.
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        drop_flag_effects_for_location(self.body, self.move_data, location, |path, s| {
            Self::update_bits(state, path, s)
        });
        if self.skip_unreachable_unwind.contains(location.block) {
            let mir::TerminatorKind::Drop { target, unwind, .. } = terminator.kind else { bug!() };
            assert_matches!(unwind, mir::UnwindAction::Cleanup(_));
            TerminatorEdges::Single(target)
        } else {
            terminator.edges()
        }
    }

    fn apply_call_return_effect(
        &mut self,
        state: &mut Self::Domain,
        _block: mir::BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            // when a call returns successfully, that means we need to set
            // the bits for that dest_place to 0 (initialized).
            on_lookup_result_bits(
                self.move_data(),
                self.move_data().rev_lookup.find(place.as_ref()),
                |mpi| {
                    state.kill(mpi);
                },
            );
        });
    }

    fn get_switch_int_data(
        &mut self,
        block: mir::BasicBlock,
        discr: &mir::Operand<'tcx>,
    ) -> Option<Self::SwitchIntData> {
        if !self.tcx.sess.opts.unstable_opts.precise_enum_drop_elaboration {
            return None;
        }

        if !self.mark_inactive_variants_as_uninit {
            return None;
        }

        MaybePlacesSwitchIntData::new(self.tcx, self.body, block, discr)
    }

    fn apply_switch_int_edge_effect(
        &mut self,
        data: &mut Self::SwitchIntData,
        state: &mut Self::Domain,
        value: SwitchTargetValue,
        targets: &mir::SwitchTargets,
    ) {
        let inactive_variants = match value {
            SwitchTargetValue::Normal(value) => InactiveVariants::Active(data.next_discr(value)),
            SwitchTargetValue::Otherwise if self.include_inactive_in_otherwise => {
                InactiveVariants::Inactives(data.variants(targets))
            }
            _ => return,
        };

        // Mark all move paths that correspond to variants other than this one as maybe
        // uninitialized (in reality, they are *definitely* uninitialized).
        drop_flag_effects::on_all_inactive_variants(
            self.move_data,
            data.enum_place,
            &inactive_variants,
            |mpi| state.gen_(mpi),
        );
    }
}

/// There can be many more `InitIndex` than there are locals in a MIR body.
/// We use a mixed bitset to avoid paying too high a memory footprint.
pub type EverInitializedPlacesDomain = MixedBitSet<InitIndex>;

impl<'tcx> Analysis<'tcx> for EverInitializedPlaces<'_, 'tcx> {
    type Domain = EverInitializedPlacesDomain;

    const NAME: &'static str = "ever_init";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = no initialized variables by default
        MixedBitSet::new_empty(self.move_data().inits.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        for arg_init in 0..body.arg_count {
            state.insert(InitIndex::new(arg_init));
        }
    }

    #[instrument(skip(self, state), level = "debug")]
    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        location: Location,
    ) {
        let move_data = self.move_data();
        let init_path_map = &move_data.init_path_map;
        let init_loc_map = &move_data.init_loc_map;
        let rev_lookup = &move_data.rev_lookup;

        debug!("initializes move_indexes {:?}", init_loc_map[location]);
        state.gen_all(init_loc_map[location].iter().copied());

        if let mir::StatementKind::StorageDead(local) = stmt.kind
            // End inits for StorageDead, so that an immutable variable can
            // be reinitialized on the next iteration of the loop.
            && let Some(move_path_index) = rev_lookup.find_local(local)
        {
            debug!("clears the ever initialized status of {:?}", init_path_map[move_path_index]);
            state.kill_all(init_path_map[move_path_index].iter().copied());
        }
    }

    #[instrument(skip(self, state, terminator), level = "debug")]
    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        let (body, move_data) = (self.body, self.move_data());
        let term = body[location.block].terminator();
        let init_loc_map = &move_data.init_loc_map;
        debug!(?term);
        debug!("initializes move_indexes {:?}", init_loc_map[location]);
        state.gen_all(
            init_loc_map[location]
                .iter()
                .filter(|init_index| {
                    move_data.inits[**init_index].kind != InitKind::NonPanicPathOnly
                })
                .copied(),
        );
        terminator.edges()
    }

    fn apply_call_return_effect(
        &mut self,
        state: &mut Self::Domain,
        block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        let move_data = self.move_data();
        let init_loc_map = &move_data.init_loc_map;

        let call_loc = self.body.terminator_loc(block);
        for init_index in &init_loc_map[call_loc] {
            state.gen_(*init_index);
        }
    }
}
