// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::attr::AttrMetaMethods;

use rustc::ty::TyCtxt;
use rustc::mir::repr::{self, Mir};

use std::io;
use std::mem;
use std::usize;

use super::MirBorrowckCtxt;
use super::gather_moves::{Location, MoveData, MovePathData, MovePathIndex, MoveOutIndex, PathMap};
use super::graphviz;
use bitslice::BitSlice; // adds set_bit/get_bit to &[usize] bitvector rep.

pub trait Dataflow {
    fn dataflow(&mut self);
}

impl<'b, 'a: 'b, 'tcx: 'a> Dataflow for MirBorrowckCtxt<'b, 'a, 'tcx> {
    fn dataflow(&mut self) {
        self.build_gen_and_kill_sets();
        self.pre_dataflow_instrumentation().unwrap();
        self.propagate();
        self.post_dataflow_instrumentation().unwrap();
    }
}

struct PropagationContext<'c, 'b: 'c, 'a: 'b, 'tcx: 'a, OnReturn>
    where OnReturn: Fn(&MoveData, &mut [usize], &repr::Lvalue)
{
    mbcx: &'c mut MirBorrowckCtxt<'b, 'a, 'tcx>,
    changed: bool,
    on_return: OnReturn
}

impl<'b, 'a: 'b, 'tcx: 'a> MirBorrowckCtxt<'b, 'a, 'tcx> {
    fn propagate(&mut self) {
        let mut temp = vec![0; self.flow_state.sets.words_per_block];
        let mut propcx = PropagationContext {
            mbcx: &mut *self,
            changed: true,
            on_return: |move_data, in_out, dest_lval| {
                let move_path_index = move_data.rev_lookup.find(dest_lval);
                on_all_children_bits(in_out,
                                     &move_data.path_map,
                                     &move_data.move_paths,
                                     move_path_index,
                                     &|in_out, mpi| {
                                         in_out.clear_bit(mpi.idx());
                                     });
            },
        };
        while propcx.changed {
            propcx.changed = false;
            propcx.reset(&mut temp);
            propcx.walk_cfg(&mut temp);
        }
    }

    fn build_gen_and_kill_sets(&mut self) {
        // First we need to build the gen- and kill-sets. The
        // gather_moves information provides a high-level mapping from
        // mir-locations to the MoveOuts (and those correspond
        // directly to gen-sets here). But we still need to figure out
        // the kill-sets.

        let move_data = &self.flow_state.operator;
        let move_paths = &move_data.move_paths;
        let loc_map = &move_data.loc_map;
        let path_map = &move_data.path_map;
        let rev_lookup = &move_data.rev_lookup;

        for bb in self.mir.all_basic_blocks() {
            let &repr::BasicBlockData { ref statements,
                                        ref terminator,
                                        is_cleanup: _ } =
                self.mir.basic_block_data(bb);

            let mut sets = self.flow_state.sets.for_block(bb.index());
            for (j, stmt) in statements.iter().enumerate() {
                let loc = Location { block: bb, index: j };
                debug!("stmt {:?} at loc {:?} moves out of move_indexes {:?}",
                       stmt, loc, &loc_map[loc]);
                for move_index in &loc_map[loc] {
                    // Every path deinitialized by a *particular move*
                    // has corresponding bit, "gen'ed" (i.e. set)
                    // here, in dataflow vector
                    zero_to_one(&mut sets.gen_set, *move_index);
                }
                match stmt.kind {
                    repr::StatementKind::Assign(ref lvalue, _) => {
                        // assigning into this `lvalue` kills all
                        // MoveOuts from it, and *also* all MoveOuts
                        // for children and associated fragment sets.
                        let move_path_index = rev_lookup.find(lvalue);

                        on_all_children_bits(sets.kill_set,
                                             path_map,
                                             move_paths,
                                             move_path_index,
                                             &|kill_set, mpi| {
                                                 kill_set.set_bit(mpi.idx());
                                             });
                    }
                }
            }

            let loc = Location { block: bb, index: statements.len() };
            debug!("terminator {:?} at loc {:?} moves out of move_indexes {:?}",
                   terminator, loc, &loc_map[loc]);
            for move_index in &loc_map[loc] {
                zero_to_one(&mut sets.gen_set, *move_index);
            }
        }

        fn zero_to_one(gen_set: &mut [usize], move_index: MoveOutIndex) {
            let retval = gen_set.set_bit(move_index.idx());
            assert!(retval);
        }
    }
}

fn on_all_children_bits<Each>(set: &mut [usize],
                              path_map: &PathMap,
                              move_paths: &MovePathData,
                              move_path_index: MovePathIndex,
                              each_child: &Each)
    where Each: Fn(&mut [usize], MoveOutIndex)
{
    // 1. invoke `each_child` callback for all moves that directly
    //    influence path for `move_path_index`
    for move_index in &path_map[move_path_index] {
        each_child(set, *move_index);
    }

    // 2. for each child of the path (that is named in this
    //    function), recur.
    //
    // (Unnamed children are irrelevant to dataflow; by
    // definition they have no associated moves.)
    let mut next_child_index = move_paths[move_path_index].first_child;
    while let Some(child_index) = next_child_index {
        on_all_children_bits(set, path_map, move_paths, child_index, each_child);
        next_child_index = move_paths[child_index].next_sibling;
    }
}

impl<'c, 'b: 'c, 'a: 'b, 'tcx: 'a, OnReturn> PropagationContext<'c, 'b, 'a, 'tcx, OnReturn>
    where OnReturn: Fn(&MoveData, &mut [usize], &repr::Lvalue)
{
    fn reset(&mut self, bits: &mut [usize]) {
        let e = if MoveData::initial_value() {usize::MAX} else {0};
        for b in bits {
            *b = e;
        }
    }

    fn walk_cfg(&mut self, in_out: &mut [usize]) {
        let &mut MirBorrowckCtxt { ref mir, ref mut flow_state, .. } = self.mbcx;
        for (idx, bb) in mir.basic_blocks.iter().enumerate() {
            {
                let sets = flow_state.sets.for_block(idx);
                debug_assert!(in_out.len() == sets.on_entry.len());
                in_out.clone_from_slice(sets.on_entry);
                bitwise(in_out, sets.gen_set, &Union);
                bitwise(in_out, sets.kill_set, &Subtract);
            }
            flow_state.propagate_bits_into_graph_successors_of(in_out,
                                                               &mut self.changed,
                                                               bb,
                                                               &self.on_return);
        }
    }
}

impl<'b, 'a: 'b, 'tcx: 'a> MirBorrowckCtxt<'b, 'a, 'tcx> {
    fn pre_dataflow_instrumentation(&self) -> io::Result<()> {
        self.if_attr_meta_name_found(
            "borrowck_graphviz_preflow",
            |this, path: &str| {
                graphviz::print_borrowck_graph_to(this, "preflow", path)
            })
    }

    fn post_dataflow_instrumentation(&self) -> io::Result<()> {
        self.if_attr_meta_name_found(
            "borrowck_graphviz_postflow",
            |this, path: &str| {
                graphviz::print_borrowck_graph_to(this, "postflow", path)
            })
    }

    fn if_attr_meta_name_found<F>(&self,
                                  name: &str,
                                  callback: F) -> io::Result<()>
        where F: for <'aa, 'bb> FnOnce(&'aa Self, &'bb str) -> io::Result<()>
    {
        for attr in self.attributes {
            if attr.check_name("rustc_mir") {
                let items = attr.meta_item_list();
                for item in items.iter().flat_map(|l| l.iter()) {
                    if item.check_name(name) {
                        if let Some(s) = item.value_str() {
                            return callback(self, &s);
                        } else {
                            self.bcx.tcx.sess.span_err(
                                item.span,
                                &format!("{} attribute requires a path", item.name()));
                        }
                    }
                }
            }
        }

        Ok(())
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

pub struct DataflowState<O: BitDenotation>
{
    /// All the sets for the analysis. (Factored into its
    /// own structure so that we can borrow it mutably
    /// on its own separate from other fields.)
    pub sets: AllSets,

    /// operator used to initialize, combine, and interpret bits.
    operator: O,
}

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
    pub fn bytes_per_block(&self) -> usize { (self.bits_per_block + 7) / 8 }
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
    pub fn on_exit_set_for(&self, block_idx: usize) -> Vec<usize> {
        let mut set: Vec<_> = self.on_entry_set_for(block_idx).iter()
            .map(|x|*x)
            .collect();
        bitwise(&mut set[..], self.gen_set_for(block_idx), &Union);
        bitwise(&mut set[..], self.kill_set_for(block_idx), &Subtract);
        return set;
    }
}

impl<O: BitDenotation> DataflowState<O> {
    fn each_bit<F>(&self, words: &[usize], mut f: F)
        where F: FnMut(usize) {
        //! Helper for iterating over the bits in a bitvector.

        for (word_index, &word) in words.iter().enumerate() {
            if word != 0 {
                let usize_bits: usize = mem::size_of::<usize>();
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
                        if bit_index >= self.sets.bits_per_block() {
                            return;
                        } else {
                            f(bit_index);
                        }
                    }
                }
            }
        }
    }

    pub fn interpret_set(&self, words: &[usize]) -> Vec<&O::Bit> {
        let mut v = Vec::new();
        self.each_bit(words, |i| {
            v.push(self.operator.interpret(i));
        });
        v
    }
}

pub trait BitwiseOperator {
    /// Joins two predecessor bits together, typically either `|` or `&`
    fn join(&self, pred1: usize, pred2: usize) -> usize;
}

/// Parameterization for the precise form of data flow that is used.
pub trait DataflowOperator : BitwiseOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn initial_value() -> bool;
}

pub trait BitDenotation: DataflowOperator {
    /// Specifies what is represented by each bit in the dataflow bitvector.
    type Bit;
    /// Size of each bivector allocated for each block in the analysis.
    fn bits_per_block(&self) -> usize;
    /// Provides the meaning of each entry in the dataflow bitvector.
    /// (Mostly intended for use for better debug instrumentation.)
    fn interpret(&self, idx: usize) -> &Self::Bit;
}

impl<D: BitDenotation> DataflowState<D> {
    pub fn new(mir: &Mir, denotation: D) -> Self {
        let bits_per_block = denotation.bits_per_block();
        let usize_bits = mem::size_of::<usize>() * 8;
        let words_per_block = (bits_per_block + usize_bits - 1) / usize_bits;
        let num_blocks = mir.basic_blocks.len();
        let num_words = num_blocks * words_per_block;

        let entry = if D::initial_value() { usize::MAX } else {0};

        let zeroes = Bits::new(0, num_words);
        let on_entry = Bits::new(entry, num_words);

        DataflowState {
            sets: AllSets {
                bits_per_block: bits_per_block,
                words_per_block: words_per_block,
                gen_sets: zeroes.clone(),
                kill_sets: zeroes,
                on_entry_sets: on_entry,
            },
            operator: denotation,
        }
    }
}

impl<D: BitDenotation> DataflowState<D> {
    /// Propagates the bits of `in_out` into all the successors of `bb`,
    /// using bitwise operator denoted by `self.operator`.
    ///
    /// For most blocks, this is entirely uniform. However, for blocks
    /// that end with a call terminator, the effect of the call on the
    /// dataflow state may depend on whether the call returned
    /// successfully or unwound. To reflect this, the `on_return`
    /// callback mutates `in_out` when propagating `in_out` via a call
    /// terminator; such mutation is performed *last*, to ensure its
    /// side-effects do not leak elsewhere (e.g. into unwind target).
    fn propagate_bits_into_graph_successors_of<OnReturn>(
        &mut self,
        in_out: &mut [usize],
        changed: &mut bool,
        bb: &repr::BasicBlockData,
        on_return: OnReturn) where OnReturn: Fn(&D, &mut [usize], &repr::Lvalue)
    {
        match bb.terminator().kind {
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
                    on_return(&self.operator, in_out, dest_lval);
                    self.propagate_bits_into_entry_set_for(in_out, changed, dest_bb);
                }
            }
        }
    }

    fn propagate_bits_into_entry_set_for(&mut self,
                                         in_out: &[usize],
                                         changed: &mut bool,
                                         bb: &repr::BasicBlock) {
        let entry_set = self.sets.for_block(bb.index()).on_entry;
        let set_changed = bitwise(entry_set, in_out, &self.operator);
        if set_changed {
            *changed = true;
        }
    }
}


impl<'a, 'tcx> DataflowState<MoveData<'tcx>> {
    pub fn new_move_analysis(mir: &Mir<'tcx>, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        let move_data = MoveData::gather_moves(mir, tcx);
        DataflowState::new(mir, move_data)
    }
}

impl<'tcx> BitwiseOperator for MoveData<'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // moves from both preds are in scope
    }
}

impl<'tcx> DataflowOperator for MoveData<'tcx> {
    #[inline]
    fn initial_value() -> bool {
        false // no loans in scope by default
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
