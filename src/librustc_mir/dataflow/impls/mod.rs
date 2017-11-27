// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Dataflow analyses are built upon some interpretation of the
//! bitvectors attached to each basic block, represented via a
//! zero-sized structure.

use rustc::ty::TyCtxt;
use rustc::mir::{self, Mir, Location};
use rustc_data_structures::bitslice::{BitwiseOperator};
use rustc_data_structures::indexed_set::{IdxSet};
use rustc_data_structures::indexed_vec::Idx;

use super::MoveDataParamEnv;
use util::elaborate_drops::DropFlagState;

use super::move_paths::{HasMoveData, MoveData, MoveOutIndex, MovePathIndex, InitIndex};
use super::move_paths::{LookupResult, InitKind};
use super::{BitDenotation, BlockSets, InitialFlow};

use super::drop_flag_effects_for_function_entry;
use super::drop_flag_effects_for_location;
use super::{on_lookup_result_bits, for_location_inits};

mod storage_liveness;

pub use self::storage_liveness::*;

#[allow(dead_code)]
pub(super) mod borrows;

/// `MaybeInitializedLvals` tracks all l-values that might be
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
/// To determine whether an l-value *must* be initialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeUninitializedLvals` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeUninitializedLvals` yields the set of
/// l-values that would require a dynamic drop-flag at that statement.
pub struct MaybeInitializedLvals<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>,
}

impl<'a, 'gcx: 'tcx, 'tcx> MaybeInitializedLvals<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>)
               -> Self
    {
        MaybeInitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'gcx, 'tcx> HasMoveData<'tcx> for MaybeInitializedLvals<'a, 'gcx, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

/// `MaybeUninitializedLvals` tracks all l-values that might be
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
/// To determine whether an l-value *must* be uninitialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeInitializedLvals` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeInitializedLvals` yields the set of
/// l-values that would require a dynamic drop-flag at that statement.
pub struct MaybeUninitializedLvals<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> MaybeUninitializedLvals<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>)
               -> Self
    {
        MaybeUninitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'gcx, 'tcx> HasMoveData<'tcx> for MaybeUninitializedLvals<'a, 'gcx, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

/// `DefinitelyInitializedLvals` tracks all l-values that are definitely
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// FIXME: Note that once flow-analysis is complete, this should be
/// the set-complement of MaybeUninitializedLvals; thus we can get rid
/// of one or the other of these two. I'm inclined to get rid of
/// MaybeUninitializedLvals, simply because the sets will tend to be
/// smaller in this analysis and thus easier for humans to process
/// when debugging.
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
/// To determine whether an l-value *may* be uninitialized at a
/// particular control-flow point, one can take the set-complement
/// of this data.
///
/// Similarly, at a given `drop` statement, the set-difference between
/// this data and `MaybeInitializedLvals` yields the set of l-values
/// that would require a dynamic drop-flag at that statement.
pub struct DefinitelyInitializedLvals<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx: 'a> DefinitelyInitializedLvals<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>)
               -> Self
    {
        DefinitelyInitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'gcx, 'tcx: 'a> HasMoveData<'tcx> for DefinitelyInitializedLvals<'a, 'gcx, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

/// `MovingOutStatements` tracks the statements that perform moves out
/// of particular l-values. More precisely, it tracks whether the
/// *effect* of such moves (namely, the uninitialization of the
/// l-value in question) can reach some point in the control-flow of
/// the function, or if that effect is "killed" by some intervening
/// operation reinitializing that l-value.
///
/// The resulting dataflow is a more enriched version of
/// `MaybeUninitializedLvals`. Both structures on their own only tell
/// you if an l-value *might* be uninitialized at a given point in the
/// control flow. But `MovingOutStatements` also includes the added
/// data of *which* particular statement causing the deinitialization
/// that the borrow checker's error message may need to report.
#[allow(dead_code)]
pub struct MovingOutStatements<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>,
}

impl<'a, 'gcx: 'tcx, 'tcx: 'a> MovingOutStatements<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>)
               -> Self
    {
        MovingOutStatements { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'gcx, 'tcx> HasMoveData<'tcx> for MovingOutStatements<'a, 'gcx, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

/// `EverInitializedLvals` tracks all l-values that might have ever been
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
pub struct EverInitializedLvals<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>,
}

impl<'a, 'gcx: 'tcx, 'tcx: 'a> EverInitializedLvals<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'gcx, 'tcx>)
               -> Self
    {
        EverInitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'gcx, 'tcx> HasMoveData<'tcx> for EverInitializedLvals<'a, 'gcx, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}


impl<'a, 'gcx, 'tcx> MaybeInitializedLvals<'a, 'gcx, 'tcx> {
    fn update_bits(sets: &mut BlockSets<MovePathIndex>, path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => sets.kill(&path),
            DropFlagState::Present => sets.gen(&path),
        }
    }
}

impl<'a, 'gcx, 'tcx> MaybeUninitializedLvals<'a, 'gcx, 'tcx> {
    fn update_bits(sets: &mut BlockSets<MovePathIndex>, path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => sets.gen(&path),
            DropFlagState::Present => sets.kill(&path),
        }
    }
}

impl<'a, 'gcx, 'tcx> DefinitelyInitializedLvals<'a, 'gcx, 'tcx> {
    fn update_bits(sets: &mut BlockSets<MovePathIndex>, path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => sets.kill(&path),
            DropFlagState::Present => sets.gen(&path),
        }
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for MaybeInitializedLvals<'a, 'gcx, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "maybe_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    fn start_block_effect(&self, entry_set: &mut IdxSet<MovePathIndex>) {
        drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                entry_set.add(&path);
            });
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<MovePathIndex>,
                        location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.mir, self.mdpe,
            location,
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<MovePathIndex>,
                         location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.mir, self.mdpe,
            location,
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn propagate_call_return(&self,
                             in_out: &mut IdxSet<MovePathIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             dest_place: &mir::Place) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 1 (initialized).
        on_lookup_result_bits(self.tcx, self.mir, self.move_data(),
                              self.move_data().rev_lookup.find(dest_place),
                              |mpi| { in_out.add(&mpi); });
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for MaybeUninitializedLvals<'a, 'gcx, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "maybe_uninit" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg places
    fn start_block_effect(&self, entry_set: &mut IdxSet<MovePathIndex>) {
        // set all bits to 1 (uninit) before gathering counterevidence
        for e in entry_set.words_mut() { *e = !0; }

        drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                entry_set.remove(&path);
            });
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<MovePathIndex>,
                        location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.mir, self.mdpe,
            location,
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<MovePathIndex>,
                         location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.mir, self.mdpe,
            location,
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn propagate_call_return(&self,
                             in_out: &mut IdxSet<MovePathIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             dest_place: &mir::Place) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 0 (initialized).
        on_lookup_result_bits(self.tcx, self.mir, self.move_data(),
                              self.move_data().rev_lookup.find(dest_place),
                              |mpi| { in_out.remove(&mpi); });
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for DefinitelyInitializedLvals<'a, 'gcx, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "definite_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg places
    fn start_block_effect(&self, entry_set: &mut IdxSet<MovePathIndex>) {
        for e in entry_set.words_mut() { *e = 0; }

        drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                entry_set.add(&path);
            });
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<MovePathIndex>,
                        location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.mir, self.mdpe,
            location,
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<MovePathIndex>,
                         location: Location)
    {
        drop_flag_effects_for_location(
            self.tcx, self.mir, self.mdpe,
            location,
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn propagate_call_return(&self,
                             in_out: &mut IdxSet<MovePathIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             dest_place: &mir::Place) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 1 (initialized).
        on_lookup_result_bits(self.tcx, self.mir, self.move_data(),
                              self.move_data().rev_lookup.find(dest_place),
                              |mpi| { in_out.add(&mpi); });
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for MovingOutStatements<'a, 'gcx, 'tcx> {
    type Idx = MoveOutIndex;
    fn name() -> &'static str { "moving_out" }
    fn bits_per_block(&self) -> usize {
        self.move_data().moves.len()
    }

    fn start_block_effect(&self, _sets: &mut IdxSet<MoveOutIndex>) {
        // no move-statements have been executed prior to function
        // execution, so this method has no effect on `_sets`.
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<MoveOutIndex>,
                        location: Location) {
        let (tcx, mir, move_data) = (self.tcx, self.mir, self.move_data());
        let stmt = &mir[location.block].statements[location.statement_index];
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;

        match stmt.kind {
            // this analysis only tries to find moves explicitly
            // written by the user, so we ignore the move-outs
            // created by `StorageDead` and at the beginning
            // of a function.
            mir::StatementKind::StorageDead(_) => {}
            _ => {
                debug!("stmt {:?} at loc {:?} moves out of move_indexes {:?}",
                       stmt, location, &loc_map[location]);
                // Every path deinitialized by a *particular move*
                // has corresponding bit, "gen'ed" (i.e. set)
                // here, in dataflow vector
                sets.gen_all_and_assert_dead(&loc_map[location]);
            }
        }

        for_location_inits(tcx, mir, move_data, location,
                           |mpi| sets.kill_all(&path_map[mpi]));
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<MoveOutIndex>,
                         location: Location)
    {
        let (tcx, mir, move_data) = (self.tcx, self.mir, self.move_data());
        let term = mir[location.block].terminator();
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;

        debug!("terminator {:?} at loc {:?} moves out of move_indexes {:?}",
               term, location, &loc_map[location]);
        sets.gen_all_and_assert_dead(&loc_map[location]);

        for_location_inits(tcx, mir, move_data, location,
                           |mpi| sets.kill_all(&path_map[mpi]));
    }

    fn propagate_call_return(&self,
                             in_out: &mut IdxSet<MoveOutIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             dest_place: &mir::Place) {
        let move_data = self.move_data();
        let bits_per_block = self.bits_per_block();

        let path_map = &move_data.path_map;
        on_lookup_result_bits(self.tcx,
                              self.mir,
                              move_data,
                              move_data.rev_lookup.find(dest_place),
                              |mpi| for moi in &path_map[mpi] {
                                  assert!(moi.index() < bits_per_block);
                                  in_out.remove(&moi);
                              });
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for EverInitializedLvals<'a, 'gcx, 'tcx> {
    type Idx = InitIndex;
    fn name() -> &'static str { "ever_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().inits.len()
    }

    fn start_block_effect(&self, entry_set: &mut IdxSet<InitIndex>) {
        for arg_init in 0..self.mir.arg_count {
            entry_set.add(&InitIndex::new(arg_init));
        }
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<InitIndex>,
                        location: Location) {
        let (_, mir, move_data) = (self.tcx, self.mir, self.move_data());
        let stmt = &mir[location.block].statements[location.statement_index];
        let init_path_map = &move_data.init_path_map;
        let init_loc_map = &move_data.init_loc_map;
        let rev_lookup = &move_data.rev_lookup;

        debug!("statement {:?} at loc {:?} initializes move_indexes {:?}",
               stmt, location, &init_loc_map[location]);
        sets.gen_all(&init_loc_map[location]);

        match stmt.kind {
            mir::StatementKind::StorageDead(local) |
            mir::StatementKind::StorageLive(local) => {
                // End inits for StorageDead and StorageLive, so that an immutable
                // variable can be reinitialized on the next iteration of the loop.
                //
                // FIXME(#46525): We *need* to do this for StorageLive as well as
                // StorageDead, because lifetimes of match bindings with guards are
                // weird - i.e. this code
                //
                // ```
                //     fn main() {
                //         match 0 {
                //             a | a
                //             if { println!("a={}", a); false } => {}
                //             _ => {}
                //         }
                //     }
                // ```
                //
                // runs the guard twice, using the same binding for `a`, and only
                // storagedeads after everything ends, so if we don't regard the
                // storagelive as killing storage, we would have a multiple assignment
                // to immutable data error.
                if let LookupResult::Exact(mpi) = rev_lookup.find(&mir::Place::Local(local)) {
                    debug!("stmt {:?} at loc {:?} clears the ever initialized status of {:?}",
                           stmt, location, &init_path_map[mpi]);
                    sets.kill_all(&init_path_map[mpi]);
                }
            }
            _ => {}
        }
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<InitIndex>,
                         location: Location)
    {
        let (mir, move_data) = (self.mir, self.move_data());
        let term = mir[location.block].terminator();
        let init_loc_map = &move_data.init_loc_map;
        debug!("terminator {:?} at loc {:?} initializes move_indexes {:?}",
               term, location, &init_loc_map[location]);
        sets.gen_all(
            init_loc_map[location].iter().filter(|init_index| {
                move_data.inits[**init_index].kind != InitKind::NonPanicPathOnly
            })
        );
    }

    fn propagate_call_return(&self,
                             in_out: &mut IdxSet<InitIndex>,
                             call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             _dest_place: &mir::Place) {
        let move_data = self.move_data();
        let bits_per_block = self.bits_per_block();
        let init_loc_map = &move_data.init_loc_map;

        let call_loc = Location {
            block: call_bb,
            statement_index: self.mir[call_bb].statements.len(),
        };
        for init_index in &init_loc_map[call_loc] {
            assert!(init_index.index() < bits_per_block);
            in_out.add(init_index);
        }
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for MaybeInitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // "maybe" means we union effects of both preds
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for MaybeUninitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // "maybe" means we union effects of both preds
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for DefinitelyInitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 & pred2 // "definitely" means we intersect effects of both preds
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for MovingOutStatements<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // moves from both preds are in scope
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for EverInitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // inits from both preds are in scope
    }
}

// The way that dataflow fixed point iteration works, you want to
// start at bottom and work your way to a fixed point. Control-flow
// merges will apply the `join` operator to each block entry's current
// state (which starts at that bottom value).
//
// This means, for propagation across the graph, that you either want
// to start at all-zeroes and then use Union as your merge when
// propagating, or you start at all-ones and then use Intersect as
// your merge when propagating.

impl<'a, 'gcx, 'tcx> InitialFlow for MaybeInitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = uninitialized
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for MaybeUninitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = initialized (start_block_effect counters this at outset)
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for DefinitelyInitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        true // bottom = initialized (start_block_effect counters this at outset)
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for MovingOutStatements<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = no loans in scope by default
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for EverInitializedLvals<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = no initialized variables by default
    }
}
