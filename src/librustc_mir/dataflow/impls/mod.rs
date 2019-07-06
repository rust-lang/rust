//! Dataflow analyses are built upon some interpretation of the
//! bitvectors attached to each basic block, represented via a
//! zero-sized structure.

use rustc::ty::TyCtxt;
use rustc::mir::{self, Body, Location};
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::Idx;

use super::MoveDataParamEnv;

use crate::util::elaborate_drops::DropFlagState;

use super::move_paths::{HasMoveData, MoveData, MovePathIndex, InitIndex, InitKind};
use super::{BitDenotation, BottomValue, GenKillSet};

use super::drop_flag_effects_for_function_entry;
use super::drop_flag_effects_for_location;
use super::on_lookup_result_bits;

mod storage_liveness;

pub use self::storage_liveness::*;

mod borrowed_locals;

pub use self::borrowed_locals::*;

pub(super) mod borrows;

/// `MaybeInitializedPlaces` tracks all places that might be
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // maybe-init:
///                                            // {}
///     let a = S; let b = S; let c; let d;    // {a, b}
///
///     if pred {
///         drop(a);                           // {   b}
///         b = S;                             // {   b}
///
///     } else {
///         drop(b);                           // {a}
///         d = S;                             // {a,       d}
///
///     }                                      // {a, b,    d}
///
///     c = S;                                 // {a, b, c, d}
/// }
/// ```
///
/// To determine whether a place *must* be initialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeUninitializedPlaces` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeUninitializedPlaces` yields the set of
/// places that would require a dynamic drop-flag at that statement.
pub struct MaybeInitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> MaybeInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        MaybeInitializedPlaces { tcx: tcx, body: body, mdpe: mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for MaybeInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
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
/// fn foo(pred: bool) {                       // maybe-uninit:
///                                            // {a, b, c, d}
///     let a = S; let b = S; let c; let d;    // {      c, d}
///
///     if pred {
///         drop(a);                           // {a,    c, d}
///         b = S;                             // {a,    c, d}
///
///     } else {
///         drop(b);                           // {   b, c, d}
///         d = S;                             // {   b, c   }
///
///     }                                      // {a, b, c, d}
///
///     c = S;                                 // {a, b,    d}
/// }
/// ```
///
/// To determine whether a place *must* be uninitialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeInitializedPlaces` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeInitializedPlaces` yields the set of
/// places that would require a dynamic drop-flag at that statement.
pub struct MaybeUninitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> MaybeUninitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        MaybeUninitializedPlaces { tcx: tcx, body: body, mdpe: mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for MaybeUninitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

/// `DefinitelyInitializedPlaces` tracks all places that are definitely
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // definite-init:
///                                            // {          }
///     let a = S; let b = S; let c; let d;    // {a, b      }
///
///     if pred {
///         drop(a);                           // {   b,     }
///         b = S;                             // {   b,     }
///
///     } else {
///         drop(b);                           // {a,        }
///         d = S;                             // {a,       d}
///
///     }                                      // {          }
///
///     c = S;                                 // {       c  }
/// }
/// ```
///
/// To determine whether a place *may* be uninitialized at a
/// particular control-flow point, one can take the set-complement
/// of this data.
///
/// Similarly, at a given `drop` statement, the set-difference between
/// this data and `MaybeInitializedPlaces` yields the set of places
/// that would require a dynamic drop-flag at that statement.
pub struct DefinitelyInitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> DefinitelyInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        DefinitelyInitializedPlaces { tcx: tcx, body: body, mdpe: mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for DefinitelyInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

/// `EverInitializedPlaces` tracks all places that might have ever been
/// initialized upon reaching a particular point in the control flow
/// for a function, without an intervening `Storage Dead`.
///
/// This dataflow is used to determine if an immutable local variable may
/// be assigned to.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // ever-init:
///                                            // {          }
///     let a = S; let b = S; let c; let d;    // {a, b      }
///
///     if pred {
///         drop(a);                           // {a, b,     }
///         b = S;                             // {a, b,     }
///
///     } else {
///         drop(b);                           // {a, b,      }
///         d = S;                             // {a, b,    d }
///
///     }                                      // {a, b,    d }
///
///     c = S;                                 // {a, b, c, d }
/// }
/// ```
pub struct EverInitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> EverInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        EverInitializedPlaces { tcx: tcx, body: body, mdpe: mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for EverInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

impl<'a, 'tcx> MaybeInitializedPlaces<'a, 'tcx> {
    fn update_bits(trans: &mut GenKillSet<MovePathIndex>,
                   path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => trans.kill(path),
            DropFlagState::Present => trans.gen(path),
        }
    }
}

impl<'a, 'tcx> MaybeUninitializedPlaces<'a, 'tcx> {
    fn update_bits(trans: &mut GenKillSet<MovePathIndex>,
                   path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => trans.gen(path),
            DropFlagState::Present => trans.kill(path),
        }
    }
}

impl<'a, 'tcx> DefinitelyInitializedPlaces<'a, 'tcx> {
    fn update_bits(trans: &mut GenKillSet<MovePathIndex>,
                   path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => trans.kill(path),
            DropFlagState::Present => trans.gen(path),
        }
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for MaybeInitializedPlaces<'a, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "maybe_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    fn start_block_effect(&self, entry_set: &mut BitSet<MovePathIndex>) {
        drop_flag_effects_for_function_entry(
            self.tcx, self.body, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                entry_set.insert(path);
            });
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Self::Idx>,
                        location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.body, self.mdpe,
            location,
            |path, s| Self::update_bits(trans, path, s)
        )
    }

    fn terminator_effect(&self,
                         trans: &mut GenKillSet<Self::Idx>,
                         location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.body, self.mdpe,
            location,
            |path, s| Self::update_bits(trans, path, s)
        )
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<MovePathIndex>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    ) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 1 (initialized).
        on_lookup_result_bits(self.tcx, self.body, self.move_data(),
                              self.move_data().rev_lookup.find(dest_place),
                              |mpi| { in_out.insert(mpi); });
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for MaybeUninitializedPlaces<'a, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "maybe_uninit" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg places
    fn start_block_effect(&self, entry_set: &mut BitSet<MovePathIndex>) {
        // set all bits to 1 (uninit) before gathering counterevidence
        assert!(self.bits_per_block() == entry_set.domain_size());
        entry_set.insert_all();

        drop_flag_effects_for_function_entry(
            self.tcx, self.body, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                entry_set.remove(path);
            });
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Self::Idx>,
                        location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.body, self.mdpe,
            location,
            |path, s| Self::update_bits(trans, path, s)
        )
    }

    fn terminator_effect(&self,
                         trans: &mut GenKillSet<Self::Idx>,
                         location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.body, self.mdpe,
            location,
            |path, s| Self::update_bits(trans, path, s)
        )
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<MovePathIndex>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    ) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 0 (initialized).
        on_lookup_result_bits(self.tcx, self.body, self.move_data(),
                              self.move_data().rev_lookup.find(dest_place),
                              |mpi| { in_out.remove(mpi); });
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for DefinitelyInitializedPlaces<'a, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "definite_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg places
    fn start_block_effect(&self, entry_set: &mut BitSet<MovePathIndex>) {
        entry_set.clear();

        drop_flag_effects_for_function_entry(
            self.tcx, self.body, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                entry_set.insert(path);
            });
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Self::Idx>,
                        location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.body, self.mdpe,
            location,
            |path, s| Self::update_bits(trans, path, s)
        )
    }

    fn terminator_effect(&self,
                         trans: &mut GenKillSet<Self::Idx>,
                         location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.body, self.mdpe,
            location,
            |path, s| Self::update_bits(trans, path, s)
        )
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<MovePathIndex>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    ) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 1 (initialized).
        on_lookup_result_bits(self.tcx, self.body, self.move_data(),
                              self.move_data().rev_lookup.find(dest_place),
                              |mpi| { in_out.insert(mpi); });
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for EverInitializedPlaces<'a, 'tcx> {
    type Idx = InitIndex;
    fn name() -> &'static str { "ever_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().inits.len()
    }

    fn start_block_effect(&self, entry_set: &mut BitSet<InitIndex>) {
        for arg_init in 0..self.body.arg_count {
            entry_set.insert(InitIndex::new(arg_init));
        }
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Self::Idx>,
                        location: Location) {
        let (_, body, move_data) = (self.tcx, self.body, self.move_data());
        let stmt = &body[location.block].statements[location.statement_index];
        let init_path_map = &move_data.init_path_map;
        let init_loc_map = &move_data.init_loc_map;
        let rev_lookup = &move_data.rev_lookup;

        debug!("statement {:?} at loc {:?} initializes move_indexes {:?}",
               stmt, location, &init_loc_map[location]);
        trans.gen_all(&init_loc_map[location]);

        match stmt.kind {
            mir::StatementKind::StorageDead(local) => {
                // End inits for StorageDead, so that an immutable variable can
                // be reinitialized on the next iteration of the loop.
                let move_path_index = rev_lookup.find_local(local);
                debug!("stmt {:?} at loc {:?} clears the ever initialized status of {:?}",
                        stmt, location, &init_path_map[move_path_index]);
                trans.kill_all(&init_path_map[move_path_index]);
            }
            _ => {}
        }
    }

    fn terminator_effect(&self,
                         trans: &mut GenKillSet<Self::Idx>,
                         location: Location)
    {
        let (body, move_data) = (self.body, self.move_data());
        let term = body[location.block].terminator();
        let init_loc_map = &move_data.init_loc_map;
        debug!("terminator {:?} at loc {:?} initializes move_indexes {:?}",
               term, location, &init_loc_map[location]);
        trans.gen_all(
            init_loc_map[location].iter().filter(|init_index| {
                move_data.inits[**init_index].kind != InitKind::NonPanicPathOnly
            })
        );
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<InitIndex>,
        call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        _dest_place: &mir::Place<'tcx>,
    ) {
        let move_data = self.move_data();
        let bits_per_block = self.bits_per_block();
        let init_loc_map = &move_data.init_loc_map;

        let call_loc = Location {
            block: call_bb,
            statement_index: self.body[call_bb].statements.len(),
        };
        for init_index in &init_loc_map[call_loc] {
            assert!(init_index.index() < bits_per_block);
            in_out.insert(*init_index);
        }
    }
}

impl<'a, 'tcx> BottomValue for MaybeInitializedPlaces<'a, 'tcx> {
    /// bottom = uninitialized
    const BOTTOM_VALUE: bool = false;
}

impl<'a, 'tcx> BottomValue for MaybeUninitializedPlaces<'a, 'tcx> {
    /// bottom = initialized (start_block_effect counters this at outset)
    const BOTTOM_VALUE: bool = false;
}

impl<'a, 'tcx> BottomValue for DefinitelyInitializedPlaces<'a, 'tcx> {
    /// bottom = initialized (start_block_effect counters this at outset)
    const BOTTOM_VALUE: bool = true;
}

impl<'a, 'tcx> BottomValue for EverInitializedPlaces<'a, 'tcx> {
    /// bottom = no initialized variables by default
    const BOTTOM_VALUE: bool = false;
}
