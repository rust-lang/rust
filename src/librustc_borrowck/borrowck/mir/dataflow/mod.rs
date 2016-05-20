// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TyCtxt;
use rustc::mir::repr::{self, Mir};

use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::path::PathBuf;
use std::usize;

use super::MirBorrowckCtxtPreDataflow;
use super::gather_moves::{Location, MoveData, MovePathIndex, MoveOutIndex};
use super::gather_moves::{MoveOut, MovePath};
use super::DropFlagState;

use bitslice::BitSlice; // adds set_bit/get_bit to &[usize] bitvector rep.

pub use self::sanity_check::sanity_check_via_rustc_peek;

mod graphviz;
mod sanity_check;

pub trait Dataflow {
    fn dataflow(&mut self);
}

impl<'a, 'tcx: 'a, BD> Dataflow for MirBorrowckCtxtPreDataflow<'a, 'tcx, BD>
    where BD: BitDenotation + DataflowOperator, BD::Bit: Debug, BD::Ctxt: HasMoveData<'tcx>
{
    fn dataflow(&mut self) {
        self.flow_state.build_sets();
        self.pre_dataflow_instrumentation().unwrap();
        self.flow_state.propagate();
        self.post_dataflow_instrumentation().unwrap();
    }
}

struct PropagationContext<'b, 'a: 'b, 'tcx: 'a, O>
    where O: 'b + BitDenotation, O::Ctxt: HasMoveData<'tcx>,
{
    builder: &'b mut DataflowAnalysis<'a, 'tcx, O>,
    changed: bool,
}

impl<'a, 'tcx: 'a, BD> DataflowAnalysis<'a, 'tcx, BD>
    where BD: BitDenotation + DataflowOperator, BD::Ctxt: HasMoveData<'tcx>
{
    fn propagate(&mut self) {
        let mut temp = vec![0; self.flow_state.sets.words_per_block];
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
            let sets = &mut self.flow_state.sets.for_block(repr::START_BLOCK.index());
            self.flow_state.operator.start_block_effect(&self.ctxt, sets);
        }

        for bb in self.mir.all_basic_blocks() {
            let &repr::BasicBlockData { ref statements,
                                        ref terminator,
                                        is_cleanup: _ } =
                self.mir.basic_block_data(bb);

            let sets = &mut self.flow_state.sets.for_block(bb.index());
            for j_stmt in 0..statements.len() {
                self.flow_state.operator.statement_effect(&self.ctxt, sets, bb, j_stmt);
            }

            if terminator.is_some() {
                let stmts_len = statements.len();
                self.flow_state.operator.terminator_effect(&self.ctxt, sets, bb, stmts_len);
            }
        }
    }
}

impl<'b, 'a: 'b, 'tcx: 'a, BD> PropagationContext<'b, 'a, 'tcx, BD>
    where BD: BitDenotation + DataflowOperator, BD::Ctxt: HasMoveData<'tcx>
{
    fn reset(&mut self, bits: &mut [usize]) {
        let e = if BD::bottom_value() {usize::MAX} else {0};
        for b in bits {
            *b = e;
        }
    }

    fn walk_cfg(&mut self, in_out: &mut [usize]) {
        let mir = self.builder.mir;
        for (bb_idx, bb_data) in mir.basic_blocks.iter().enumerate() {
            let builder = &mut self.builder;
            {
                let sets = builder.flow_state.sets.for_block(bb_idx);
                debug_assert!(in_out.len() == sets.on_entry.len());
                in_out.clone_from_slice(sets.on_entry);
                bitwise(in_out, sets.gen_set, &Union);
                bitwise(in_out, sets.kill_set, &Subtract);
            }
            builder.propagate_bits_into_graph_successors_of(in_out,
                                                            &mut self.changed,
                                                            (repr::BasicBlock::new(bb_idx), bb_data));
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
    where BD: BitDenotation, BD::Bit: Debug, BD::Ctxt: HasMoveData<'tcx>
{
    fn pre_dataflow_instrumentation(&self) -> io::Result<()> {
        if let Some(ref path_str) = self.print_preflow_to {
            let path = dataflow_path(BD::name(), "preflow", path_str);
            graphviz::print_borrowck_graph_to(self, &path)
        } else {
            Ok(())
        }
    }

    fn post_dataflow_instrumentation(&self) -> io::Result<()> {
        if let Some(ref path_str) = self.print_postflow_to {
            let path = dataflow_path(BD::name(), "postflow", path_str);
            graphviz::print_borrowck_graph_to(self, &path)
        } else{
            Ok(())
        }
    }
}

/// Maps each block to a set of bits
#[derive(Clone, Debug)]
struct Bits {
    bits: Vec<usize>,
}

impl Bits {
    fn new(init_word: usize, num_words: usize) -> Self {
        Bits { bits: vec![init_word; num_words] }
    }
}

pub trait HasMoveData<'tcx> {
    fn move_data(&self) -> &MoveData<'tcx>;
}

impl<'tcx> HasMoveData<'tcx> for MoveData<'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> { self }
}
impl<'tcx, A, B> HasMoveData<'tcx> for (A, B, MoveData<'tcx>) {
    fn move_data(&self) -> &MoveData<'tcx> { &self.2 }
}

pub struct DataflowAnalysis<'a, 'tcx: 'a, O>
    where O: BitDenotation, O::Ctxt: HasMoveData<'tcx>
{
    flow_state: DataflowState<O>,
    ctxt: O::Ctxt,
    mir: &'a Mir<'tcx>,
}

impl<'a, 'tcx: 'a, O> DataflowAnalysis<'a, 'tcx, O>
    where O: BitDenotation, O::Ctxt: HasMoveData<'tcx>
{
    pub fn results(self) -> (O::Ctxt, DataflowResults<O>) {
        (self.ctxt, DataflowResults(self.flow_state))
    }

    pub fn mir(&self) -> &'a Mir<'tcx> { self.mir }
}

#[derive(Debug)]
pub struct DataflowResults<O: BitDenotation>(DataflowState<O>);

// FIXME: This type shouldn't be public, but the graphviz::MirWithFlowState trait
// references it in a method signature. Look into using `pub(crate)` to address this.
#[derive(Debug)]
pub struct DataflowState<O: BitDenotation>
{
    /// All the sets for the analysis. (Factored into its
    /// own structure so that we can borrow it mutably
    /// on its own separate from other fields.)
    pub sets: AllSets,

    /// operator used to initialize, combine, and interpret bits.
    operator: O,
}

#[derive(Debug)]
pub struct AllSets {
    /// Analysis bitwidth for each block.
    bits_per_block: usize,

    /// Number of words associated with each block entry
    /// equal to bits_per_block / usize::BITS, rounded up.
    words_per_block: usize,

    /// For each block, bits generated by executing the statements in
    /// the block. (For comparison, the Terminator for each block is
    /// handled in a flow-specific manner during propagation.)
    gen_sets: Bits,

    /// For each block, bits killed by executing the statements in the
    /// block. (For comparison, the Terminator for each block is
    /// handled in a flow-specific manner during propagation.)
    kill_sets: Bits,

    /// For each block, bits valid on entry to the block.
    on_entry_sets: Bits,
}

pub struct BlockSets<'a> {
    on_entry: &'a mut [usize],
    gen_set: &'a mut [usize],
    kill_set: &'a mut [usize],
}

impl AllSets {
    pub fn bits_per_block(&self) -> usize { self.bits_per_block }
    pub fn for_block(&mut self, block_idx: usize) -> BlockSets {
        let offset = self.words_per_block * block_idx;
        let range = offset..(offset + self.words_per_block);
        BlockSets {
            on_entry: &mut self.on_entry_sets.bits[range.clone()],
            gen_set: &mut self.gen_sets.bits[range.clone()],
            kill_set: &mut self.kill_sets.bits[range],
        }
    }

    fn lookup_set_for<'a>(&self, sets: &'a Bits, block_idx: usize) -> &'a [usize] {
        let offset = self.words_per_block * block_idx;
        &sets.bits[offset..(offset + self.words_per_block)]
    }
    pub fn gen_set_for(&self, block_idx: usize) -> &[usize] {
        self.lookup_set_for(&self.gen_sets, block_idx)
    }
    pub fn kill_set_for(&self, block_idx: usize) -> &[usize] {
        self.lookup_set_for(&self.kill_sets, block_idx)
    }
    pub fn on_entry_set_for(&self, block_idx: usize) -> &[usize] {
        self.lookup_set_for(&self.on_entry_sets, block_idx)
    }
}

impl<O: BitDenotation> DataflowState<O> {
    fn each_bit<F>(&self, ctxt: &O::Ctxt, words: &[usize], mut f: F)
        where F: FnMut(usize) {
        //! Helper for iterating over the bits in a bitvector.

        let bits_per_block = self.operator.bits_per_block(ctxt);
        let usize_bits: usize = mem::size_of::<usize>() * 8;

        for (word_index, &word) in words.iter().enumerate() {
            if word != 0 {
                let base_index = word_index * usize_bits;
                for offset in 0..usize_bits {
                    let bit = 1 << offset;
                    if (word & bit) != 0 {
                        // NB: we round up the total number of bits
                        // that we store in any given bit set so that
                        // it is an even multiple of usize::BITS. This
                        // means that there may be some stray bits at
                        // the end that do not correspond to any
                        // actual value; that's why we first check
                        // that we are in range of bits_per_block.
                        let bit_index = base_index + offset as usize;
                        if bit_index >= bits_per_block {
                            return;
                        } else {
                            f(bit_index);
                        }
                    }
                }
            }
        }
    }

    pub fn interpret_set<'c>(&self, ctxt: &'c O::Ctxt, words: &[usize]) -> Vec<&'c O::Bit> {
        let mut v = Vec::new();
        self.each_bit(ctxt, words, |i| {
            v.push(self.operator.interpret(ctxt, i));
        });
        v
    }
}

pub trait BitwiseOperator {
    /// Applies some bit-operation pointwise to each of the bits in the two inputs.
    fn join(&self, pred1: usize, pred2: usize) -> usize;
}

/// Parameterization for the precise form of data flow that is used.
pub trait DataflowOperator: BitwiseOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn bottom_value() -> bool;
}

pub trait BitDenotation {
    /// Specifies what is represented by each bit in the dataflow bitvector.
    type Bit;

    /// Specifies what, if any, separate context needs to be supplied for methods below.
    type Ctxt;

    /// A name describing the dataflow analysis that this
    /// BitDenotation is supporting.  The name should be something
    /// suitable for plugging in as part of a filename e.g. avoid
    /// space-characters or other things that tend to look bad on a
    /// file system, like slashes or periods. It is also better for
    /// the name to be reasonably short, again because it will be
    /// plugged into a filename.
    fn name() -> &'static str;

    /// Size of each bitvector allocated for each block in the analysis.
    fn bits_per_block(&self, &Self::Ctxt) -> usize;

    /// Provides the meaning of each entry in the dataflow bitvector.
    /// (Mostly intended for use for better debug instrumentation.)
    fn interpret<'a>(&self, &'a Self::Ctxt, idx: usize) -> &'a Self::Bit;

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects that have been
    /// established *prior* to entering the start block.
    ///
    /// (For example, establishing the call arguments.)
    ///
    /// (Typically this should only modify `sets.on_entry`, since the
    /// gen and kill sets should reflect the effects of *executing*
    /// the start block itself.)
    fn start_block_effect(&self, ctxt: &Self::Ctxt, sets: &mut BlockSets);

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating statement.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" represnting the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The statement here is `idx_stmt.1`; `idx_stmt.0` is just
    /// an identifying index: namely, the index of the statement
    /// in the basic block.
    fn statement_effect(&self,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        idx_stmt: usize);

    /// Mutates the block-sets (the flow sets for the given
    /// basic block) according to the effects of evaluating
    /// the terminator.
    ///
    /// This is used, in particular, for building up the
    /// "transfer-function" represnting the overall-effect of the
    /// block, represented via GEN and KILL sets.
    ///
    /// The terminator here is `idx_term.1`; `idx_term.0` is just an
    /// identifying index: namely, the number of statements in `bb`
    /// itself.
    ///
    /// The effects applied here cannot depend on which branch the
    /// terminator took.
    fn terminator_effect(&self,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
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
    /// Note: as a historical artifact, this currently takes as input
    /// the *entire* packed collection of bitvectors in `in_out`.  We
    /// might want to look into narrowing that to something more
    /// specific, just to make the interface more self-documenting.
    fn propagate_call_return(&self,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             call_bb: repr::BasicBlock,
                             dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue);
}

impl<'a, 'tcx: 'a, D> DataflowAnalysis<'a, 'tcx, D>
    where D: BitDenotation + DataflowOperator, D::Ctxt: HasMoveData<'tcx>
{
    pub fn new(_tcx: TyCtxt<'a, 'tcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               ctxt: D::Ctxt,
               denotation: D) -> Self {
        let bits_per_block = denotation.bits_per_block(&ctxt);
        let usize_bits = mem::size_of::<usize>() * 8;
        let words_per_block = (bits_per_block + usize_bits - 1) / usize_bits;
        let num_blocks = mir.basic_blocks.len();
        let num_words = num_blocks * words_per_block;

        let entry = if D::bottom_value() { usize::MAX } else {0};

        let zeroes = Bits::new(0, num_words);
        let on_entry = Bits::new(entry, num_words);

        DataflowAnalysis { flow_state: DataflowState {
            sets: AllSets {
                bits_per_block: bits_per_block,
                words_per_block: words_per_block,
                gen_sets: zeroes.clone(),
                kill_sets: zeroes,
                on_entry_sets: on_entry,
            },
            operator: denotation,
        },
                           ctxt: ctxt,
                           mir: mir,
        }

    }
}

impl<'a, 'tcx: 'a, D> DataflowAnalysis<'a, 'tcx, D>
    where D: BitDenotation + DataflowOperator, D::Ctxt: HasMoveData<'tcx>
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
        in_out: &mut [usize],
        changed: &mut bool,
        (bb, bb_data): (repr::BasicBlock, &repr::BasicBlockData))
    {
        match bb_data.terminator().kind {
            repr::TerminatorKind::Return |
            repr::TerminatorKind::Resume => {}
            repr::TerminatorKind::Goto { ref target } |
            repr::TerminatorKind::Drop { ref target, value: _, unwind: None } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, target);
            }
            repr::TerminatorKind::Drop { ref target, value: _, unwind: Some(ref unwind) } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, target);
                self.propagate_bits_into_entry_set_for(in_out, changed, unwind);
            }
            repr::TerminatorKind::If { ref targets, .. } => {
                self.propagate_bits_into_entry_set_for(in_out, changed, &targets.0);
                self.propagate_bits_into_entry_set_for(in_out, changed, &targets.1);
            }
            repr::TerminatorKind::Switch { ref targets, .. } |
            repr::TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets {
                    self.propagate_bits_into_entry_set_for(in_out, changed, target);
                }
            }
            repr::TerminatorKind::Call { ref cleanup, ref destination, func: _, args: _ } => {
                if let Some(ref unwind) = *cleanup {
                    self.propagate_bits_into_entry_set_for(in_out, changed, unwind);
                }
                if let Some((ref dest_lval, ref dest_bb)) = *destination {
                    // N.B.: This must be done *last*, after all other
                    // propagation, as documented in comment above.
                    self.flow_state.operator.propagate_call_return(
                        &self.ctxt, in_out, bb, *dest_bb, dest_lval);
                    self.propagate_bits_into_entry_set_for(in_out, changed, dest_bb);
                }
            }
        }
    }

    fn propagate_bits_into_entry_set_for(&mut self,
                                         in_out: &[usize],
                                         changed: &mut bool,
                                         bb: &repr::BasicBlock) {
        let entry_set = self.flow_state.sets.for_block(bb.index()).on_entry;
        let set_changed = bitwise(entry_set, in_out, &self.flow_state.operator);
        if set_changed {
            *changed = true;
        }
    }
}

// Dataflow analyses are built upon some interpretation of the
// bitvectors attached to each basic block, represented via a
// zero-sized structure.
//
// Note on PhantomData: Each interpretation will need to instantiate
// the `Bit` and `Ctxt` associated types, and in this case, those
// associated types need an associated lifetime `'tcx`. The
// interpretive structures are zero-sized, so they all need to carry a
// `PhantomData` representing how the structures relate to the `'tcx`
// lifetime.
//
// But, since all of the uses of `'tcx` are solely via instances of
// `Ctxt` that are passed into the `BitDenotation` methods, we can
// consistently use a `PhantomData` that is just a function over a
// `&Ctxt` (== `&MoveData<'tcx>).

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
#[derive(Debug, Default)]
pub struct MaybeInitializedLvals<'a, 'tcx: 'a> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<Fn(&'a MoveData<'tcx>, TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>)>
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
#[derive(Debug, Default)]
pub struct MaybeUninitializedLvals<'a, 'tcx: 'a> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<Fn(&'a MoveData<'tcx>, TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>)>
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
#[derive(Debug, Default)]
pub struct DefinitelyInitializedLvals<'a, 'tcx: 'a> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<Fn(&'a MoveData<'tcx>, TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>)>
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
/// that the borrow checker's error meessage may need to report.
#[derive(Debug, Default)]
pub struct MovingOutStatements<'a, 'tcx: 'a> {
    // See "Note on PhantomData" above.
    phantom: PhantomData<Fn(&'a MoveData<'tcx>, TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>)>
}

impl<'a, 'tcx> BitDenotation for MovingOutStatements<'a, 'tcx> {
    type Bit = MoveOut;
    type Ctxt = (TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>, MoveData<'tcx>);
    fn name() -> &'static str { "moving_out" }
    fn bits_per_block(&self, ctxt: &Self::Ctxt) -> usize {
        ctxt.2.moves.len()
    }
    fn interpret<'c>(&self, ctxt: &'c Self::Ctxt, idx: usize) -> &'c Self::Bit {
        &ctxt.2.moves[idx]
    }
    fn start_block_effect(&self,_move_data: &Self::Ctxt, _sets: &mut BlockSets) {
        // no move-statements have been executed prior to function
        // execution, so this method has no effect on `_sets`.
    }
    fn statement_effect(&self,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        idx: usize) {
        let &(tcx, mir, ref move_data) = ctxt;
        let stmt = &mir.basic_block_data(bb).statements[idx];
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;

        let loc = Location { block: bb, index: idx };
        debug!("stmt {:?} at loc {:?} moves out of move_indexes {:?}",
               stmt, loc, &loc_map[loc]);
        for move_index in &loc_map[loc] {
            // Every path deinitialized by a *particular move*
            // has corresponding bit, "gen'ed" (i.e. set)
            // here, in dataflow vector
            zero_to_one(&mut sets.gen_set, *move_index);
        }
        let bits_per_block = self.bits_per_block(ctxt);
        match stmt.kind {
            repr::StatementKind::Assign(ref lvalue, _) => {
                // assigning into this `lvalue` kills all
                // MoveOuts from it, and *also* all MoveOuts
                // for children and associated fragment sets.
                let move_path_index = rev_lookup.find(lvalue);

                sets.kill_set.set_bit(move_path_index.idx());
                super::on_all_children_bits(tcx,
                                            mir,
                                            move_data,
                                            move_path_index,
                                            |mpi| for moi in &path_map[mpi] {
                                                assert!(moi.idx() < bits_per_block);
                                                sets.kill_set.set_bit(moi.idx());
                                            });
            }
        }
    }

    fn terminator_effect(&self,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         statements_len: usize)
    {
        let &(_tcx, mir, ref move_data) = ctxt;
        let term = mir.basic_block_data(bb).terminator.as_ref().unwrap();
        let loc_map = &move_data.loc_map;
        let loc = Location { block: bb, index: statements_len };
        debug!("terminator {:?} at loc {:?} moves out of move_indexes {:?}",
               term, loc, &loc_map[loc]);
        let bits_per_block = self.bits_per_block(ctxt);
        for move_index in &loc_map[loc] {
            assert!(move_index.idx() < bits_per_block);
            zero_to_one(&mut sets.gen_set, *move_index);
        }
    }

    fn propagate_call_return(&self,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             _call_bb: repr::BasicBlock,
                             _dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        let move_data = &ctxt.2;
        let move_path_index = move_data.rev_lookup.find(dest_lval);
        let bits_per_block = self.bits_per_block(ctxt);

        in_out.clear_bit(move_path_index.idx());
        let path_map = &move_data.path_map;
        super::on_all_children_bits(ctxt.0,
                                    ctxt.1,
                                    move_data,
                                    move_path_index,
                                    |mpi| for moi in &path_map[mpi] {
                                        assert!(moi.idx() < bits_per_block);
                                        in_out.clear_bit(moi.idx());
                                    });
    }
}

impl<'a, 'tcx> MaybeInitializedLvals<'a, 'tcx> {
    fn update_bits(sets: &mut BlockSets, path: MovePathIndex,
                   state: super::DropFlagState)
    {
        match state {
            DropFlagState::Absent => {
                sets.gen_set.clear_bit(path.idx());
                sets.kill_set.set_bit(path.idx());
            }
            DropFlagState::Present => {
                sets.gen_set.set_bit(path.idx());
                sets.kill_set.clear_bit(path.idx());
            }
        }
    }
}

impl<'a, 'tcx> MaybeUninitializedLvals<'a, 'tcx> {
    fn update_bits(sets: &mut BlockSets, path: MovePathIndex,
                   state: super::DropFlagState)
    {
        match state {
            DropFlagState::Absent => {
                sets.gen_set.set_bit(path.idx());
                sets.kill_set.clear_bit(path.idx());
            }
            DropFlagState::Present => {
                sets.gen_set.clear_bit(path.idx());
                sets.kill_set.set_bit(path.idx());
            }
        }
    }
}

impl<'a, 'tcx> DefinitelyInitializedLvals<'a, 'tcx> {
    fn update_bits(sets: &mut BlockSets, path: MovePathIndex,
                   state: super::DropFlagState)
    {
        match state {
            DropFlagState::Absent => {
                sets.gen_set.clear_bit(path.idx());
                sets.kill_set.set_bit(path.idx());
            }
            DropFlagState::Present => {
                sets.gen_set.set_bit(path.idx());
                sets.kill_set.clear_bit(path.idx());
            }
        }
    }
}

impl<'a, 'tcx> BitDenotation for MaybeInitializedLvals<'a, 'tcx> {
    type Bit = MovePath<'tcx>;
    type Ctxt = (TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>, MoveData<'tcx>);
    fn name() -> &'static str { "maybe_init" }
    fn bits_per_block(&self, ctxt: &Self::Ctxt) -> usize {
        ctxt.2.move_paths.len()
    }
    fn interpret<'c>(&self, ctxt: &'c Self::Ctxt, idx: usize) -> &'c Self::Bit {
        &ctxt.2.move_paths[MovePathIndex::new(idx)]
    }
    fn start_block_effect(&self, ctxt: &Self::Ctxt, sets: &mut BlockSets)
    {
        super::drop_flag_effects_for_function_entry(
            ctxt.0, ctxt.1, &ctxt.2,
            |path, s| {
                assert!(s == DropFlagState::Present);
                sets.on_entry.set_bit(path.idx());
            });
    }

    fn statement_effect(&self,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        idx: usize)
    {
        super::drop_flag_effects_for_location(
            ctxt.0, ctxt.1, &ctxt.2,
            Location { block: bb, index: idx },
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn terminator_effect(&self,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         statements_len: usize)
    {
        super::drop_flag_effects_for_location(
            ctxt.0, ctxt.1, &ctxt.2,
            Location { block: bb, index: statements_len },
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn propagate_call_return(&self,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             _call_bb: repr::BasicBlock,
                             _dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 1 (initialized).
        let move_data = &ctxt.2;
        let move_path_index = move_data.rev_lookup.find(dest_lval);
        super::on_all_children_bits(
            ctxt.0, ctxt.1, &ctxt.2,
            move_path_index,
            |mpi| { in_out.set_bit(mpi.idx()); }
        );
    }
}

impl<'a, 'tcx> BitDenotation for MaybeUninitializedLvals<'a, 'tcx> {
    type Bit = MovePath<'tcx>;
    type Ctxt = (TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>, MoveData<'tcx>);
    fn name() -> &'static str { "maybe_uninit" }
    fn bits_per_block(&self, ctxt: &Self::Ctxt) -> usize {
        ctxt.2.move_paths.len()
    }
    fn interpret<'c>(&self, ctxt: &'c Self::Ctxt, idx: usize) -> &'c Self::Bit {
        &ctxt.2.move_paths[MovePathIndex::new(idx)]
    }

    // sets on_entry bits for Arg lvalues
    fn start_block_effect(&self, ctxt: &Self::Ctxt, sets: &mut BlockSets) {
        // set all bits to 1 (uninit) before gathering counterevidence
        for e in &mut sets.on_entry[..] { *e = !0; }

        super::drop_flag_effects_for_function_entry(
            ctxt.0, ctxt.1, &ctxt.2,
            |path, s| {
                assert!(s == DropFlagState::Present);
                sets.on_entry.clear_bit(path.idx());
            });
    }

    fn statement_effect(&self,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        idx: usize)
    {
        super::drop_flag_effects_for_location(
            ctxt.0, ctxt.1, &ctxt.2,
            Location { block: bb, index: idx },
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn terminator_effect(&self,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         statements_len: usize)
    {
        super::drop_flag_effects_for_location(
            ctxt.0, ctxt.1, &ctxt.2,
            Location { block: bb, index: statements_len },
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn propagate_call_return(&self,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             _call_bb: repr::BasicBlock,
                             _dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 1 (initialized).
        let move_path_index = ctxt.2.rev_lookup.find(dest_lval);
        super::on_all_children_bits(
            ctxt.0, ctxt.1, &ctxt.2,
            move_path_index,
            |mpi| { in_out.clear_bit(mpi.idx()); }
        );
    }
}

impl<'a, 'tcx> BitDenotation for DefinitelyInitializedLvals<'a, 'tcx> {
    type Bit = MovePath<'tcx>;
    type Ctxt = (TyCtxt<'a, 'tcx, 'tcx>, &'a Mir<'tcx>, MoveData<'tcx>);
    fn name() -> &'static str { "definite_init" }
    fn bits_per_block(&self, ctxt: &Self::Ctxt) -> usize {
        ctxt.2.move_paths.len()
    }
    fn interpret<'c>(&self, ctxt: &'c Self::Ctxt, idx: usize) -> &'c Self::Bit {
        &ctxt.2.move_paths[MovePathIndex::new(idx)]
    }

    // sets on_entry bits for Arg lvalues
    fn start_block_effect(&self, ctxt: &Self::Ctxt, sets: &mut BlockSets) {
        for e in &mut sets.on_entry[..] { *e = 0; }

        super::drop_flag_effects_for_function_entry(
            ctxt.0, ctxt.1, &ctxt.2,
            |path, s| {
                assert!(s == DropFlagState::Present);
                sets.on_entry.set_bit(path.idx());
            });
    }

    fn statement_effect(&self,
                        ctxt: &Self::Ctxt,
                        sets: &mut BlockSets,
                        bb: repr::BasicBlock,
                        idx: usize)
    {
        super::drop_flag_effects_for_location(
            ctxt.0, ctxt.1, &ctxt.2,
            Location { block: bb, index: idx },
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn terminator_effect(&self,
                         ctxt: &Self::Ctxt,
                         sets: &mut BlockSets,
                         bb: repr::BasicBlock,
                         statements_len: usize)
    {
        super::drop_flag_effects_for_location(
            ctxt.0, ctxt.1, &ctxt.2,
            Location { block: bb, index: statements_len },
            |path, s| Self::update_bits(sets, path, s)
        )
    }

    fn propagate_call_return(&self,
                             ctxt: &Self::Ctxt,
                             in_out: &mut [usize],
                             _call_bb: repr::BasicBlock,
                             _dest_bb: repr::BasicBlock,
                             dest_lval: &repr::Lvalue) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_lval to 1 (initialized).
        let move_path_index = ctxt.2.rev_lookup.find(dest_lval);
        super::on_all_children_bits(
            ctxt.0, ctxt.1, &ctxt.2,
            move_path_index,
            |mpi| { in_out.set_bit(mpi.idx()); }
        );
    }
}

fn zero_to_one(bitvec: &mut [usize], move_index: MoveOutIndex) {
    let retval = bitvec.set_bit(move_index.idx());
    assert!(retval);
}

impl<'a, 'tcx> BitwiseOperator for MovingOutStatements<'a, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // moves from both preds are in scope
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

impl<'a, 'tcx> DataflowOperator for MovingOutStatements<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = no loans in scope by default
    }
}

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
        true // bottom = initialized
    }
}

#[inline]
fn bitwise<Op:BitwiseOperator>(out_vec: &mut [usize],
                               in_vec: &[usize],
                               op: &Op) -> bool {
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elt, in_elt) in out_vec.iter_mut().zip(in_vec) {
        let old_val = *out_elt;
        let new_val = op.join(old_val, *in_elt);
        *out_elt = new_val;
        changed |= old_val != new_val;
    }
    changed
}

struct Union;
impl BitwiseOperator for Union {
    fn join(&self, a: usize, b: usize) -> usize { a | b }
}
struct Subtract;
impl BitwiseOperator for Subtract {
    fn join(&self, a: usize, b: usize) -> usize { a & !b }
}
