// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_set::{IdxSet, IdxSetBuf};
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::bitslice::{bitwise, BitwiseOperator};

use rustc::ty::TyCtxt;
use rustc::mir::{self, Mir};

use std::fmt::Debug;
use std::io;
use std::mem;
use std::path::PathBuf;
use std::usize;

use super::MirBorrowckCtxtPreDataflow;

pub use self::sanity_check::sanity_check_via_rustc_peek;
pub use self::impls::{MaybeInitializedLvals, MaybeUninitializedLvals};
pub use self::impls::{DefinitelyInitializedLvals, MovingOutStatements};

mod graphviz;
mod sanity_check;
mod impls;

pub trait Dataflow<BD: BitDenotation> {
    fn dataflow<P>(&mut self, p: P) where P: Fn(&BD, BD::Idx) -> &Debug;
}

impl<'a, 'tcx: 'a, BD> Dataflow<BD> for MirBorrowckCtxtPreDataflow<'a, 'tcx, BD>
    where BD: BitDenotation + DataflowOperator
{
    fn dataflow<P>(&mut self, p: P) where P: Fn(&BD, BD::Idx) -> &Debug {
        self.flow_state.build_sets();
        self.pre_dataflow_instrumentation(|c,i| p(c,i)).unwrap();
        self.flow_state.propagate();
        self.post_dataflow_instrumentation(|c,i| p(c,i)).unwrap();
    }
}

struct PropagationContext<'b, 'a: 'b, 'tcx: 'a, O>
    where O: 'b + BitDenotation
{
    builder: &'b mut DataflowAnalysis<'a, 'tcx, O>,
    changed: bool,
}

impl<'a, 'tcx: 'a, BD> DataflowAnalysis<'a, 'tcx, BD>
    where BD: BitDenotation + DataflowOperator
{
    fn propagate(&mut self) {
        let mut temp = IdxSetBuf::new_empty(self.flow_state.sets.bits_per_block);
        let mut propcx = PropagationContext {
            builder: self,
            changed: true,
        };
        while propcx.changed {
            propcx.changed = false;
            propcx.reset(&mut temp);
            propcx.walk_cfg(&mut temp);
        }
    }

    fn build_sets(&mut self) {
        // First we need to build the entry-, gen- and kill-sets. The
        // gather_moves information provides a high-level mapping from
        // mir-locations to the MoveOuts (and those correspond
        // directly to gen-sets here). But we still need to figure out
        // the kill-sets.

        {
            let sets = &mut self.flow_state.sets.for_block(mir::START_BLOCK.index());
            self.flow_state.operator.start_block_effect(sets);
        }

        for (bb, data) in self.mir.basic_blocks().iter_enumerated() {
            let &mir::BasicBlockData { ref statements, ref terminator, is_cleanup: _ } = data;

            let sets = &mut self.flow_state.sets.for_block(bb.index());
            for j_stmt in 0..statements.len() {
                self.flow_state.operator.statement_effect(sets, bb, j_stmt);
            }

            if terminator.is_some() {
                let stmts_len = statements.len();
                self.flow_state.operator.terminator_effect(sets, bb, stmts_len);
            }
        }
    }
}

impl<'b, 'a: 'b, 'tcx: 'a, BD> PropagationContext<'b, 'a, 'tcx, BD>
    where BD: BitDenotation + DataflowOperator
{
    fn reset(&mut self, bits: &mut IdxSet<BD::Idx>) {
        let e = if BD::bottom_value() {!0} else {0};
        for b in bits.words_mut() {
            *b = e;
        }
    }

    fn walk_cfg(&mut self, in_out: &mut IdxSet<BD::Idx>) {
        let mir = self.builder.mir;
        for (bb_idx, bb_data) in mir.basic_blocks().iter().enumerate() {
            let builder = &mut self.builder;
            {
                let sets = builder.flow_state.sets.for_block(bb_idx);
                debug_assert!(in_out.words().len() == sets.on_entry.words().len());
                in_out.clone_from(sets.on_entry);
                in_out.union(sets.gen_set);
                in_out.subtract(sets.kill_set);
            }
            builder.propagate_bits_into_graph_successors_of(
                in_out, &mut self.changed, (mir::BasicBlock::new(bb_idx), bb_data));
        }
    }
}

fn dataflow_path(context: &str, prepost: &str, path: &str) -> PathBuf {
    format!("{}_{}", context, prepost);
    let mut path = PathBuf::from(path);
    let new_file_name = {
        let orig_file_name = path.file_name().unwrap().to_str().unwrap();
        format!("{}_{}", context, orig_file_name)
    };
    path.set_file_name(new_file_name);
    path
}

impl<'a, 'tcx: 'a, BD> MirBorrowckCtxtPreDataflow<'a, 'tcx, BD>
    where BD: BitDenotation
{
    fn pre_dataflow_instrumentation<P>(&self, p: P) -> io::Result<()>
        where P: Fn(&BD, BD::Idx) -> &Debug
    {
        if let Some(ref path_str) = self.print_preflow_to {
            let path = dataflow_path(BD::name(), "preflow", path_str);
            graphviz::print_borrowck_graph_to(self, &path, p)
        } else {
            Ok(())
        }
    }

    fn post_dataflow_instrumentation<P>(&self, p: P) -> io::Result<()>
        where P: Fn(&BD, BD::Idx) -> &Debug
    {
        if let Some(ref path_str) = self.print_postflow_to {
            let path = dataflow_path(BD::name(), "postflow", path_str);
            graphviz::print_borrowck_graph_to(self, &path, p)
        } else{
            Ok(())
        }
    }
}

/// Maps each block to a set of bits
#[derive(Debug)]
struct Bits<E:Idx> {
    bits: IdxSetBuf<E>,
}

impl<E:Idx> Clone for Bits<E> {
    fn clone(&self) -> Self { Bits { bits: self.bits.clone() } }
}

impl<E:Idx> Bits<E> {
    fn new(bits: IdxSetBuf<E>) -> Self {
        Bits { bits: bits }
    }
}

pub struct DataflowAnalysis<'a, 'tcx: 'a, O>
    where O: BitDenotation
{
    flow_state: DataflowState<O>,
    mir: &'a Mir<'tcx>,
}

impl<'a, 'tcx: 'a, O> DataflowAnalysis<'a, 'tcx, O>
    where O: BitDenotation
{
    pub fn results(self) -> DataflowResults<O> {
        DataflowResults(self.flow_state)
    }

    pub fn mir(&self) -> &'a Mir<'tcx> { self.mir }
}

pub struct DataflowResults<O>(DataflowState<O>) where O: BitDenotation;

impl<O: BitDenotation> DataflowResults<O> {
    pub fn sets(&self) -> &AllSets<O::Idx> {
        &self.0.sets
    }
}

// FIXME: This type shouldn't be public, but the graphviz::MirWithFlowState trait
// references it in a method signature. Look into using `pub(crate)` to address this.
pub struct DataflowState<O: BitDenotation>
{
    /// All the sets for the analysis. (Factored into its
    /// own structure so that we can borrow it mutably
    /// on its own separate from other fields.)
    pub sets: AllSets<O::Idx>,

    /// operator used to initialize, combine, and interpret bits.
    operator: O,
}

#[derive(Debug)]
pub struct AllSets<E: Idx> {
    /// Analysis bitwidth for each block.
    bits_per_block: usize,

    /// Number of words associated with each block entry
    /// equal to bits_per_block / usize::BITS, rounded up.
    words_per_block: usize,

    /// For each block, bits generated by executing the statements in
    /// the block. (For comparison, the Terminator for each block is
    /// handled in a flow-specific manner during propagation.)
    gen_sets: Bits<E>,

    /// For each block, bits killed by executing the statements in the
    /// block. (For comparison, the Terminator for each block is
    /// handled in a flow-specific manner during propagation.)
    kill_sets: Bits<E>,

    /// For each block, bits valid on entry to the block.
    on_entry_sets: Bits<E>,
}

pub struct BlockSets<'a, E: Idx> {
    on_entry: &'a mut IdxSet<E>,
    gen_set: &'a mut IdxSet<E>,
    kill_set: &'a mut IdxSet<E>,
}

impl<'a, E:Idx> BlockSets<'a, E> {
    fn gen(&mut self, e: &E) {
        self.gen_set.add(e);
        self.kill_set.remove(e);
    }
    fn kill(&mut self, e: &E) {
        self.gen_set.remove(e);
        self.kill_set.add(e);
    }
}

impl<E:Idx> AllSets<E> {
    pub fn bits_per_block(&self) -> usize { self.bits_per_block }
    pub fn for_block(&mut self, block_idx: usize) -> BlockSets<E> {
        let offset = self.words_per_block * block_idx;
        let range = E::new(offset)..E::new(offset + self.words_per_block);
        BlockSets {
            on_entry: self.on_entry_sets.bits.range_mut(&range),
            gen_set: self.gen_sets.bits.range_mut(&range),
            kill_set: self.kill_sets.bits.range_mut(&range),
        }
    }

    fn lookup_set_for<'a>(&self, sets: &'a Bits<E>, block_idx: usize) -> &'a IdxSet<E> {
        let offset = self.words_per_block * block_idx;
        let range = E::new(offset)..E::new(offset + self.words_per_block);
        sets.bits.range(&range)
    }
    pub fn gen_set_for(&self, block_idx: usize) -> &IdxSet<E> {
        self.lookup_set_for(&self.gen_sets, block_idx)
    }
    pub fn kill_set_for(&self, block_idx: usize) -> &IdxSet<E> {
        self.lookup_set_for(&self.kill_sets, block_idx)
    }
    pub fn on_entry_set_for(&self, block_idx: usize) -> &IdxSet<E> {
        self.lookup_set_for(&self.on_entry_sets, block_idx)
    }
}

/// Parameterization for the precise form of data flow that is used.
pub trait DataflowOperator: BitwiseOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn bottom_value() -> bool;
}

pub trait BitDenotation {
    /// Specifies what index type is used to access the bitvector.
    type Idx: Idx;

    /// A name describing the dataflow analysis that this
    /// BitDenotation is supporting.  The name should be something
    /// suitable for plugging in as part of a filename e.g. avoid
    /// space-characters or other things that tend to look bad on a
    /// file system, like slashes or periods. It is also better for
    /// the name to be reasonably short, again because it will be
    /// plugged into a filename.
    fn name() -> &'static str;

    /// Size of each bitvector allocated for each block in the analysis.
    fn bits_per_block(&self) -> usize;

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects that have been
    /// established *prior* to entering the start block.
    ///
    /// (For example, establishing the call arguments.)
    ///
    /// (Typically this should only modify `sets.on_entry`, since the
    /// gen and kill sets should reflect the effects of *executing*
    /// the start block itself.)
    fn start_block_effect(&self, sets: &mut BlockSets<Self::Idx>);

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating statement.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" represnting the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The statement is identified as `bb_data[idx_stmt]`, where
    /// `bb_data` is the sequence of statements identifed by `bb` in
    /// the MIR.
    fn statement_effect(&self,
                        sets: &mut BlockSets<Self::Idx>,
                        bb: mir::BasicBlock,
                        idx_stmt: usize);

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating
    /// the terminator.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" represnting the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The effects applied here cannot depend on which branch the
    /// terminator took.
    fn terminator_effect(&self,
                         sets: &mut BlockSets<Self::Idx>,
                         bb: mir::BasicBlock,
                         idx_term: usize);

    /// Mutates the block-sets according to the (flow-dependent)
    /// effect of a successful return from a Call terminator.
    ///
    /// If basic-block BB_x ends with a call-instruction that, upon
    /// successful return, flows to BB_y, then this method will be
    /// called on the exit flow-state of BB_x in order to set up the
    /// entry flow-state of BB_y.
    ///
    /// This is used, in particular, as a special case during the
    /// "propagate" loop where all of the basic blocks are repeatedly
    /// visited. Since the effects of a Call terminator are
    /// flow-dependent, the current MIR cannot encode them via just
    /// GEN and KILL sets attached to the block, and so instead we add
    /// this extra machinery to represent the flow-dependent effect.
    ///
    /// FIXME: Right now this is a bit of a wart in the API. It might
    /// be better to represent this as an additional gen- and
    /// kill-sets associated with each edge coming out of the basic
    /// block.
    fn propagate_call_return(&self,
                             in_out: &mut IdxSet<Self::Idx>,
                             call_bb: mir::BasicBlock,
                             dest_bb: mir::BasicBlock,
                             dest_lval: &mir::Lvalue);
}

impl<'a, 'tcx: 'a, D> DataflowAnalysis<'a, 'tcx, D>
    where D: BitDenotation + DataflowOperator
{
    pub fn new(_tcx: TyCtxt<'a, 'tcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               denotation: D) -> Self {
        let bits_per_block = denotation.bits_per_block();
        let usize_bits = mem::size_of::<usize>() * 8;
        let words_per_block = (bits_per_block + usize_bits - 1) / usize_bits;

        // (now rounded up to multiple of word size)
        let bits_per_block = words_per_block * usize_bits;

        let num_blocks = mir.basic_blocks().len();
        let num_overall = num_blocks * bits_per_block;

        let zeroes = Bits::new(IdxSetBuf::new_empty(num_overall));
        let on_entry = Bits::new(if D::bottom_value() {
            IdxSetBuf::new_filled(num_overall)
        } else {
            IdxSetBuf::new_empty(num_overall)
        });

        DataflowAnalysis {
            mir: mir,
            flow_state: DataflowState {
                sets: AllSets {
                    bits_per_block: bits_per_block,
                    words_per_block: words_per_block,
                    gen_sets: zeroes.clone(),
                    kill_sets: zeroes,
                    on_entry_sets: on_entry,
                },
                operator: denotation,
            },
        }

    }
}

impl<'a, 'tcx: 'a, D> DataflowAnalysis<'a, 'tcx, D>
    where D: BitDenotation + DataflowOperator
{
    /// Propagates the bits of `in_out` into all the successors of `bb`,
    /// using bitwise operator denoted by `self.operator`.
    ///
    /// For most blocks, this is entirely uniform. However, for blocks
    /// that end with a call terminator, the effect of the call on the
    /// dataflow state may depend on whether the call returned
    /// successfully or unwound.
    ///
    /// To reflect this, the `propagate_call_return` method of the
    /// `BitDenotation` mutates `in_out` when propagating `in_out` via
    /// a call terminator; such mutation is performed *last*, to
    /// ensure its side-effects do not leak elsewhere (e.g. into
    /// unwind target).
    fn propagate_bits_into_graph_successors_of(
        &mut self,
        in_out: &mut IdxSet<D::Idx>,
        changed: &mut bool,
        (bb, bb_data): (mir::BasicBlock, &mir::BasicBlockData))
    {
        match bb_data.terminator().kind {
            mir::TerminatorKind::Return |
            mir::TerminatorKind::Resume |
            mir::TerminatorKind::Unreachable => {}
            mir::TerminatorKind::Goto { ref target } |
            mir::TerminatorKind::Assert { ref target, cleanup: None, .. } |
            mir::TerminatorKind::Drop { ref target, location: _, unwind: None } |
            mir::TerminatorKind::DropAndReplace {
                ref target, value: _, location: _, unwind: None
            } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, target);
            }
            mir::TerminatorKind::Assert { ref target, cleanup: Some(ref unwind), .. } |
            mir::TerminatorKind::Drop { ref target, location: _, unwind: Some(ref unwind) } |
            mir::TerminatorKind::DropAndReplace {
                ref target, value: _, location: _, unwind: Some(ref unwind)
            } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, target);
                self.propagate_bits_into_entry_set_for(in_out, changed, unwind);
            }
            mir::TerminatorKind::If { ref targets, .. } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, &targets.0);
                self.propagate_bits_into_entry_set_for(in_out, changed, &targets.1);
            }
            mir::TerminatorKind::Switch { ref targets, .. } |
            mir::TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets {
                    self.propagate_bits_into_entry_set_for(in_out, changed, target);
                }
            }
            mir::TerminatorKind::Call { ref cleanup, ref destination, func: _, args: _ } => {
                if let Some(ref unwind) = *cleanup {
                    self.propagate_bits_into_entry_set_for(in_out, changed, unwind);
                }
                if let Some((ref dest_lval, ref dest_bb)) = *destination {
                    // N.B.: This must be done *last*, after all other
                    // propagation, as documented in comment above.
                    self.flow_state.operator.propagate_call_return(
                        in_out, bb, *dest_bb, dest_lval);
                    self.propagate_bits_into_entry_set_for(in_out, changed, dest_bb);
                }
            }
        }
    }

    fn propagate_bits_into_entry_set_for(&mut self,
                                         in_out: &IdxSet<D::Idx>,
                                         changed: &mut bool,
                                         bb: &mir::BasicBlock) {
        let entry_set = self.flow_state.sets.for_block(bb.index()).on_entry;
        let set_changed = bitwise(entry_set.words_mut(),
                                  in_out.words(),
                                  &self.flow_state.operator);
        if set_changed {
            *changed = true;
        }
    }
}
