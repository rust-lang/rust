// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Hook into libgraphviz for rendering dataflow graphs for MIR.

use syntax::ast::NodeId;
use rustc::mir::{BasicBlock, Mir};
use rustc_data_structures::bitslice::bits_to_string;
use rustc_data_structures::indexed_set::{IdxSet};
use rustc_data_structures::indexed_vec::Idx;

use dot;
use dot::IntoCow;

use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::path::Path;

use super::super::MirBorrowckCtxtPreDataflow;
use super::{BitDenotation, DataflowState};

impl<O: BitDenotation> DataflowState<O> {
    fn each_bit<F>(&self, words: &IdxSet<O::Idx>, mut f: F)
        where F: FnMut(O::Idx) {
        //! Helper for iterating over the bits in a bitvector.

        let bits_per_block = self.operator.bits_per_block();
        let usize_bits: usize = mem::size_of::<usize>() * 8;

        for (word_index, &word) in words.words().iter().enumerate() {
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
                            f(O::Idx::new(bit_index));
                        }
                    }
                }
            }
        }
    }

    pub fn interpret_set<'c, P>(&self,
                                o: &'c O,
                                words: &IdxSet<O::Idx>,
                                render_idx: &P)
                                -> Vec<&'c Debug>
        where P: Fn(&O, O::Idx) -> &Debug
    {
        let mut v = Vec::new();
        self.each_bit(words, |i| {
            v.push(render_idx(o, i));
        });
        v
    }
}

pub trait MirWithFlowState<'tcx> {
    type BD: BitDenotation;
    fn node_id(&self) -> NodeId;
    fn mir(&self) -> &Mir<'tcx>;
    fn flow_state(&self) -> &DataflowState<Self::BD>;
}

impl<'a, 'tcx: 'a, BD> MirWithFlowState<'tcx> for MirBorrowckCtxtPreDataflow<'a, 'tcx, BD>
    where 'tcx: 'a, BD: BitDenotation
{
    type BD = BD;
    fn node_id(&self) -> NodeId { self.node_id }
    fn mir(&self) -> &Mir<'tcx> { self.flow_state.mir() }
    fn flow_state(&self) -> &DataflowState<Self::BD> { &self.flow_state.flow_state }
}

struct Graph<'a, 'tcx, MWF:'a, P> where
    MWF: MirWithFlowState<'tcx>
{
    mbcx: &'a MWF,
    phantom: PhantomData<&'tcx ()>,
    render_idx: P,
}

pub fn print_borrowck_graph_to<'a, 'tcx, BD, P>(
    mbcx: &MirBorrowckCtxtPreDataflow<'a, 'tcx, BD>,
    path: &Path,
    render_idx: P)
    -> io::Result<()>
    where BD: BitDenotation,
          P: Fn(&BD, BD::Idx) -> &Debug
{
    let g = Graph { mbcx: mbcx, phantom: PhantomData, render_idx: render_idx };
    let mut v = Vec::new();
    dot::render(&g, &mut v)?;
    debug!("print_borrowck_graph_to path: {} node_id: {}",
           path.display(), mbcx.node_id);
    File::create(path).and_then(|mut f| f.write_all(&v))
}

pub type Node = BasicBlock;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Edge { source: BasicBlock, index: usize }

fn outgoing(mir: &Mir, bb: BasicBlock) -> Vec<Edge> {
    let succ_len = mir[bb].terminator().successors().len();
    (0..succ_len).map(|index| Edge { source: bb, index: index}).collect()
}

impl<'a, 'tcx, MWF, P> dot::Labeller<'a> for Graph<'a, 'tcx, MWF, P>
    where MWF: MirWithFlowState<'tcx>,
          P: for <'b> Fn(&'b MWF::BD, <MWF::BD as BitDenotation>::Idx) -> &'b Debug,
{
    type Node = Node;
    type Edge = Edge;
    fn graph_id(&self) -> dot::Id {
        dot::Id::new(format!("graph_for_node_{}",
                             self.mbcx.node_id()))
            .unwrap()
    }

    fn node_id(&self, n: &Node) -> dot::Id {
        dot::Id::new(format!("bb_{}", n.index()))
            .unwrap()
    }

    fn node_label(&self, n: &Node) -> dot::LabelText {
        // A standard MIR label, as generated by write_node_label, is
        // presented in a single column in a table.
        //
        // The code below does a bunch of formatting work to format a
        // node (i.e. MIR basic-block) label with extra
        // dataflow-enriched information.  In particular, the goal is
        // to add extra columns that present the three dataflow
        // bitvectors, and the data those bitvectors represent.
        //
        // It presents it in the following format (where I am
        // presenting the table rendering via ASCII art, one line per
        // row of the table, and a chunk size of 3 rather than 5):
        //
        // ------  -----------------------  ------------  --------------------
        //                    [e1, e3, e4]
        //             [e8, e9] "= ENTRY:"  <ENTRY-BITS>
        // ------  -----------------------  ------------  --------------------
        // Left
        // Most
        // Column
        // Is
        // Just
        // Normal
        // Series
        // Of
        // MIR
        // Stmts
        // ------  -----------------------  ------------  --------------------
        //           [g1, g4, g5] "= GEN:"  <GEN-BITS>
        // ------  -----------------------  ------------  --------------------
        //                         "KILL:"  <KILL-BITS>   "=" [k1, k3, k8]
        //                                                [k9]
        // ------  -----------------------  ------------  --------------------
        //
        // (In addition, the added dataflow is rendered with a colored
        // background just so it will stand out compared to the
        // statements.)
        let mut v = Vec::new();
        let i = n.index();
        let chunk_size = 5;
        const BG_FLOWCONTENT: &'static str = r#"bgcolor="pink""#;
        const ALIGN_RIGHT: &'static str = r#"align="right""#;
        const FACE_MONOSPACE: &'static str = r#"FACE="Courier""#;
        fn chunked_present_left<W:io::Write>(w: &mut W,
                                             interpreted: &[&Debug],
                                             chunk_size: usize)
                                             -> io::Result<()>
        {
            // This function may emit a sequence of <tr>'s, but it
            // always finishes with an (unfinished)
            // <tr><td></td><td>
            //
            // Thus, after being called, one should finish both the
            // pending <td> as well as the <tr> itself.
            let mut seen_one = false;
            for c in interpreted.chunks(chunk_size) {
                if seen_one {
                    // if not the first row, finish off the previous row
                    write!(w, "</td><td></td><td></td></tr>")?;
                }
                write!(w, "<tr><td></td><td {bg} {align}>{objs:?}",
                       bg = BG_FLOWCONTENT,
                       align = ALIGN_RIGHT,
                       objs = c)?;
                seen_one = true;
            }
            if !seen_one {
                write!(w, "<tr><td></td><td {bg} {align}>[]",
                       bg = BG_FLOWCONTENT,
                       align = ALIGN_RIGHT)?;
            }
            Ok(())
        }
        ::rustc_mir::graphviz::write_node_label(
            *n, self.mbcx.mir(), &mut v, 4,
            |w| {
                let flow = self.mbcx.flow_state();
                let entry_interp = flow.interpret_set(&flow.operator,
                                                      flow.sets.on_entry_set_for(i),
                                                      &self.render_idx);
                chunked_present_left(w, &entry_interp[..], chunk_size)?;
                let bits_per_block = flow.sets.bits_per_block();
                let entry = flow.sets.on_entry_set_for(i);
                debug!("entry set for i={i} bits_per_block: {bpb} entry: {e:?} interp: {ei:?}",
                       i=i, e=entry, bpb=bits_per_block, ei=entry_interp);
                write!(w, "= ENTRY:</td><td {bg}><FONT {face}>{entrybits:?}</FONT></td>\
                                        <td></td></tr>",
                       bg = BG_FLOWCONTENT,
                       face = FACE_MONOSPACE,
                       entrybits=bits_to_string(entry.words(), bits_per_block))
            },
            |w| {
                let flow = self.mbcx.flow_state();
                let gen_interp =
                    flow.interpret_set(&flow.operator, flow.sets.gen_set_for(i), &self.render_idx);
                let kill_interp =
                    flow.interpret_set(&flow.operator, flow.sets.kill_set_for(i), &self.render_idx);
                chunked_present_left(w, &gen_interp[..], chunk_size)?;
                let bits_per_block = flow.sets.bits_per_block();
                {
                    let gen = flow.sets.gen_set_for(i);
                    debug!("gen set for i={i} bits_per_block: {bpb} gen: {g:?} interp: {gi:?}",
                           i=i, g=gen, bpb=bits_per_block, gi=gen_interp);
                    write!(w, " = GEN:</td><td {bg}><FONT {face}>{genbits:?}</FONT></td>\
                                           <td></td></tr>",
                           bg = BG_FLOWCONTENT,
                           face = FACE_MONOSPACE,
                           genbits=bits_to_string(gen.words(), bits_per_block))?;
                }

                {
                    let kill = flow.sets.kill_set_for(i);
                    debug!("kill set for i={i} bits_per_block: {bpb} kill: {k:?} interp: {ki:?}",
                           i=i, k=kill, bpb=bits_per_block, ki=kill_interp);
                    write!(w, "<tr><td></td><td {bg} {align}>KILL:</td>\
                                            <td {bg}><FONT {face}>{killbits:?}</FONT></td>",
                           bg = BG_FLOWCONTENT,
                           align = ALIGN_RIGHT,
                           face = FACE_MONOSPACE,
                           killbits=bits_to_string(kill.words(), bits_per_block))?;
                }

                // (chunked_present_right)
                let mut seen_one = false;
                for k in kill_interp.chunks(chunk_size) {
                    if !seen_one {
                        // continuation of row; this is fourth <td>
                        write!(w, "<td {bg}>= {kill:?}</td></tr>",
                               bg = BG_FLOWCONTENT,
                               kill=k)?;
                    } else {
                        // new row, with indent of three <td>'s
                        write!(w, "<tr><td></td><td></td><td></td><td {bg}>{kill:?}</td></tr>",
                               bg = BG_FLOWCONTENT,
                               kill=k)?;
                    }
                    seen_one = true;
                }
                if !seen_one {
                    write!(w, "<td {bg}>= []</td></tr>",
                           bg = BG_FLOWCONTENT)?;
                }

                Ok(())
            })
            .unwrap();
        dot::LabelText::html(String::from_utf8(v).unwrap())
    }

    fn node_shape(&self, _n: &Node) -> Option<dot::LabelText> {
        Some(dot::LabelText::label("none"))
    }
}

impl<'a, 'tcx, MWF, P> dot::GraphWalk<'a> for Graph<'a, 'tcx, MWF, P>
    where MWF: MirWithFlowState<'tcx>
{
    type Node = Node;
    type Edge = Edge;
    fn nodes(&self) -> dot::Nodes<Node> {
        self.mbcx.mir()
            .basic_blocks()
            .indices()
            .collect::<Vec<_>>()
            .into_cow()
    }

    fn edges(&self) -> dot::Edges<Edge> {
        let mir = self.mbcx.mir();
        // base initial capacity on assumption every block has at
        // least one outgoing edge (Which should be true for all
        // blocks but one, the exit-block).
        let mut edges = Vec::with_capacity(mir.basic_blocks().len());
        for bb in mir.basic_blocks().indices() {
            let outgoing = outgoing(mir, bb);
            edges.extend(outgoing.into_iter());
        }
        edges.into_cow()
    }

    fn source(&self, edge: &Edge) -> Node {
        edge.source
    }

    fn target(&self, edge: &Edge) -> Node {
        let mir = self.mbcx.mir();
        mir[edge.source].terminator().successors()[edge.index]
    }
}
