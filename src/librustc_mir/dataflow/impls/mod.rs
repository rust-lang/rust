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

use super::MoveDataParamEnv;
use util::elaborate_drops::DropFlagState;

use super::move_paths::{HasMoveData, MoveData, MovePathIndex};
use super::{BitDenotation, BlockSets, DataflowOperator};

use super::drop_flag_effects_for_function_entry;
use super::drop_flag_effects_for_location;
use super::on_lookup_result_bits;

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
pub struct MaybeInitializedLvals<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx: 'a> MaybeInitializedLvals<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'tcx>)
               -> Self
    {
        MaybeInitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'tcx: 'a> HasMoveData<'tcx> for MaybeInitializedLvals<'a, 'tcx> {
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
pub struct MaybeUninitializedLvals<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx: 'a> MaybeUninitializedLvals<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'tcx>)
               -> Self
    {
        MaybeUninitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'tcx: 'a> HasMoveData<'tcx> for MaybeUninitializedLvals<'a, 'tcx> {
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
pub struct DefinitelyInitializedLvals<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx: 'a> DefinitelyInitializedLvals<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               mdpe: &'a MoveDataParamEnv<'tcx>)
               -> Self
    {
        DefinitelyInitializedLvals { tcx: tcx, mir: mir, mdpe: mdpe }
    }
}

impl<'a, 'tcx: 'a> HasMoveData<'tcx> for DefinitelyInitializedLvals<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { &self.mdpe.move_data }
}

impl<'a, 'tcx> MaybeInitializedLvals<'a, 'tcx> {
    fn update_bits(sets: &mut BlockSets<MovePathIndex>, path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => sets.kill(&path),
            DropFlagState::Present => sets.gen(&path),
        }
    }
}

impl<'a, 'tcx> MaybeUninitializedLvals<'a, 'tcx> {
    fn update_bits(sets: &mut BlockSets<MovePathIndex>, path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => sets.gen(&path),
            DropFlagState::Present => sets.kill(&path),
        }
    }
}

impl<'a, 'tcx> DefinitelyInitializedLvals<'a, 'tcx> {
    fn update_bits(sets: &mut BlockSets<MovePathIndex>, path: MovePathIndex,
                   state: DropFlagState)
    {
        match state {
            DropFlagState::Absent => sets.kill(&path),
            DropFlagState::Present => sets.gen(&path),
        }
    }
}

impl<'a, 'tcx> BitDenotation for MaybeInitializedLvals<'a, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "maybe_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    fn start_block_effect(&self, sets: &mut BlockSets<MovePathIndex>)
    {
        drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                sets.on_entry.add(&path);
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
                             dest_lval: &mir::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 1 (initialized).
        on_lookup_result_bits(self.tcx, self.mir, self.move_data(),
                              self.move_data().rev_lookup.find(dest_lval),
                              |mpi| { in_out.add(&mpi); });
    }
}

impl<'a, 'tcx> BitDenotation for MaybeUninitializedLvals<'a, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "maybe_uninit" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg lvalues
    fn start_block_effect(&self, sets: &mut BlockSets<MovePathIndex>) {
        // set all bits to 1 (uninit) before gathering counterevidence
        for e in sets.on_entry.words_mut() { *e = !0; }

        drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                sets.on_entry.remove(&path);
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
                             dest_lval: &mir::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 0 (initialized).
        on_lookup_result_bits(self.tcx, self.mir, self.move_data(),
                              self.move_data().rev_lookup.find(dest_lval),
                              |mpi| { in_out.remove(&mpi); });
    }
}

impl<'a, 'tcx> BitDenotation for DefinitelyInitializedLvals<'a, 'tcx> {
    type Idx = MovePathIndex;
    fn name() -> &'static str { "definite_init" }
    fn bits_per_block(&self) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg lvalues
    fn start_block_effect(&self, sets: &mut BlockSets<MovePathIndex>) {
        for e in sets.on_entry.words_mut() { *e = 0; }

        drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.mdpe,
            |path, s| {
                assert!(s == DropFlagState::Present);
                sets.on_entry.add(&path);
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
                             dest_lval: &mir::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 1 (initialized).
        on_lookup_result_bits(self.tcx, self.mir, self.move_data(),
                              self.move_data().rev_lookup.find(dest_lval),
                              |mpi| { in_out.add(&mpi); });
    }
}

impl<'a, 'tcx> BitwiseOperator for MaybeInitializedLvals<'a, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // "maybe" means we union effects of both preds
    }
}

impl<'a, 'tcx> BitwiseOperator for MaybeUninitializedLvals<'a, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // "maybe" means we union effects of both preds
    }
}

impl<'a, 'tcx> BitwiseOperator for DefinitelyInitializedLvals<'a, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 & pred2 // "definitely" means we intersect effects of both preds
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

impl<'a, 'tcx> DataflowOperator for MaybeInitializedLvals<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = uninitialized
    }
}

impl<'a, 'tcx> DataflowOperator for MaybeUninitializedLvals<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = initialized (start_block_effect counters this at outset)
    }
}

impl<'a, 'tcx> DataflowOperator for DefinitelyInitializedLvals<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        true // bottom = initialized (start_block_effect counters this at outset)
    }
}
