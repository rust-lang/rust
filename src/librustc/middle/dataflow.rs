// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! A module for propagating forward dataflow information. The analysis
//! assumes that the items to be propagated can be represented as bits
//! and thus uses bitvectors. Your job is simply to specify the so-called
//! GEN and KILL bits for each expression.

use cfg;
use cfg::CFGIndex;
use ty::TyCtxt;
use std::io;
use std::mem;
use std::usize;
use syntax::ast;
use syntax::print::pp;
use syntax::print::pprust::PrintState;
use util::nodemap::NodeMap;
use hir;
use hir::intravisit::{self, IdRange};
use hir::print as pprust;


#[derive(Copy, Clone, Debug)]
pub enum EntryOrExit {
    Entry,
    Exit,
}

#[derive(Clone)]
pub struct DataFlowContext<'a, 'tcx: 'a, O> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,

    /// a name for the analysis using this dataflow instance
    analysis_name: &'static str,

    /// the data flow operator
    oper: O,

    /// number of bits to propagate per id
    bits_per_id: usize,

    /// number of words we will use to store bits_per_id.
    /// equal to bits_per_id/usize::BITS rounded up.
    words_per_id: usize,

    // mapping from node to cfg node index
    // FIXME (#6298): Shouldn't this go with CFG?
    nodeid_to_index: NodeMap<Vec<CFGIndex>>,

    // Bit sets per cfg node.  The following three fields (`gens`, `kills`,
    // and `on_entry`) all have the same structure. For each id in
    // `id_range`, there is a range of words equal to `words_per_id`.
    // So, to access the bits for any given id, you take a slice of
    // the full vector (see the method `compute_id_range()`).

    /// bits generated as we exit the cfg node. Updated by `add_gen()`.
    gens: Vec<usize>,

    /// bits killed as we exit the cfg node, or non-locally jump over
    /// it. Updated by `add_kill(KillFrom::ScopeEnd)`.
    scope_kills: Vec<usize>,

    /// bits killed as we exit the cfg node directly; if it is jumped
    /// over, e.g. via `break`, the kills are not reflected in the
    /// jump's effects. Updated by `add_kill(KillFrom::Execution)`.
    action_kills: Vec<usize>,

    /// bits that are valid on entry to the cfg node. Updated by
    /// `propagate()`.
    on_entry: Vec<usize>,
}

pub trait BitwiseOperator {
    /// Joins two predecessor bits together, typically either `|` or `&`
    fn join(&self, succ: usize, pred: usize) -> usize;
}

/// Parameterization for the precise form of data flow that is used.
pub trait DataFlowOperator : BitwiseOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn initial_value(&self) -> bool;
}

struct PropagationContext<'a, 'b: 'a, 'tcx: 'b, O: 'a> {
    dfcx: &'a mut DataFlowContext<'b, 'tcx, O>,
    changed: bool
}

fn get_cfg_indices<'a>(id: ast::NodeId, index: &'a NodeMap<Vec<CFGIndex>>) -> &'a [CFGIndex] {
    let opt_indices = index.get(&id);
    opt_indices.map(|v| &v[..]).unwrap_or(&[])
}

impl<'a, 'tcx, O:DataFlowOperator> DataFlowContext<'a, 'tcx, O> {
    fn has_bitset_for_nodeid(&self, n: ast::NodeId) -> bool {
        assert!(n != ast::DUMMY_NODE_ID);
        self.nodeid_to_index.contains_key(&n)
    }
}

impl<'a, 'tcx, O:DataFlowOperator> pprust::PpAnn for DataFlowContext<'a, 'tcx, O> {
    fn nested(&self, state: &mut pprust::State, nested: pprust::Nested) -> io::Result<()> {
        pprust::PpAnn::nested(&self.tcx.map, state, nested)
    }
    fn pre(&self,
           ps: &mut pprust::State,
           node: pprust::AnnNode) -> io::Result<()> {
        let id = match node {
            pprust::NodeName(_) => ast::CRATE_NODE_ID,
            pprust::NodeExpr(expr) => expr.id,
            pprust::NodeBlock(blk) => blk.id,
            pprust::NodeItem(_) | pprust::NodeSubItem(_) => ast::CRATE_NODE_ID,
            pprust::NodePat(pat) => pat.id
        };

        if !self.has_bitset_for_nodeid(id) {
            return Ok(());
        }

        assert!(self.bits_per_id > 0);
        let indices = get_cfg_indices(id, &self.nodeid_to_index);
        for &cfgidx in indices {
            let (start, end) = self.compute_id_range(cfgidx);
            let on_entry = &self.on_entry[start.. end];
            let entry_str = bits_to_string(on_entry);

            let gens = &self.gens[start.. end];
            let gens_str = if gens.iter().any(|&u| u != 0) {
                format!(" gen: {}", bits_to_string(gens))
            } else {
                "".to_string()
            };

            let action_kills = &self.action_kills[start .. end];
            let action_kills_str = if action_kills.iter().any(|&u| u != 0) {
                format!(" action_kill: {}", bits_to_string(action_kills))
            } else {
                "".to_string()
            };

            let scope_kills = &self.scope_kills[start .. end];
            let scope_kills_str = if scope_kills.iter().any(|&u| u != 0) {
                format!(" scope_kill: {}", bits_to_string(scope_kills))
            } else {
                "".to_string()
            };

            ps.synth_comment(
                format!("id {}: {}{}{}{}", id, entry_str,
                        gens_str, action_kills_str, scope_kills_str))?;
            pp::space(&mut ps.s)?;
        }
        Ok(())
    }
}

fn build_nodeid_to_index(body: Option<&hir::Body>,
                         cfg: &cfg::CFG) -> NodeMap<Vec<CFGIndex>> {
    let mut index = NodeMap();

    // FIXME (#6298): Would it be better to fold formals from decl
    // into cfg itself?  i.e. introduce a fn-based flow-graph in
    // addition to the current block-based flow-graph, rather than
    // have to put traversals like this here?
    if let Some(body) = body {
        add_entries_from_fn_body(&mut index, body, cfg.entry);
    }

    cfg.graph.each_node(|node_idx, node| {
        if let cfg::CFGNodeData::AST(id) = node.data {
            index.entry(id).or_insert(vec![]).push(node_idx);
        }
        true
    });

    return index;

    /// Add mappings from the ast nodes for the formal bindings to
    /// the entry-node in the graph.
    fn add_entries_from_fn_body(index: &mut NodeMap<Vec<CFGIndex>>,
                                body: &hir::Body,
                                entry: CFGIndex) {
        use hir::intravisit::Visitor;

        struct Formals<'a> {
            entry: CFGIndex,
            index: &'a mut NodeMap<Vec<CFGIndex>>,
        }
        let mut formals = Formals { entry: entry, index: index };
        for arg in &body.arguments {
            formals.visit_pat(&arg.pat);
        }
        impl<'a, 'v> Visitor<'v> for Formals<'a> {
            fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'v> {
                intravisit::NestedVisitorMap::None
            }

            fn visit_pat(&mut self, p: &hir::Pat) {
                self.index.entry(p.id).or_insert(vec![]).push(self.entry);
                intravisit::walk_pat(self, p)
            }
        }
    }
}

/// Flag used by `add_kill` to indicate whether the provided kill
/// takes effect only when control flows directly through the node in
/// question, or if the kill's effect is associated with any
/// control-flow directly through or indirectly over the node.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum KillFrom {
    /// A `ScopeEnd` kill is one that takes effect when any control
    /// flow goes over the node. A kill associated with the end of the
    /// scope of a variable declaration `let x;` is an example of a
    /// `ScopeEnd` kill.
    ScopeEnd,

    /// An `Execution` kill is one that takes effect only when control
    /// flow goes through the node to completion. A kill associated
    /// with an assignment statement `x = expr;` is an example of an
    /// `Execution` kill.
    Execution,
}

impl<'a, 'tcx, O:DataFlowOperator> DataFlowContext<'a, 'tcx, O> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               analysis_name: &'static str,
               body: Option<&hir::Body>,
               cfg: &cfg::CFG,
               oper: O,
               id_range: IdRange,
               bits_per_id: usize) -> DataFlowContext<'a, 'tcx, O> {
        let usize_bits = mem::size_of::<usize>() * 8;
        let words_per_id = (bits_per_id + usize_bits - 1) / usize_bits;
        let num_nodes = cfg.graph.all_nodes().len();

        debug!("DataFlowContext::new(analysis_name: {}, id_range={:?}, \
                                     bits_per_id={}, words_per_id={}) \
                                     num_nodes: {}",
               analysis_name, id_range, bits_per_id, words_per_id,
               num_nodes);

        let entry = if oper.initial_value() { usize::MAX } else {0};

        let zeroes = vec![0; num_nodes * words_per_id];
        let gens = zeroes.clone();
        let kills1 = zeroes.clone();
        let kills2 = zeroes;
        let on_entry = vec![entry; num_nodes * words_per_id];

        let nodeid_to_index = build_nodeid_to_index(body, cfg);

        DataFlowContext {
            tcx: tcx,
            analysis_name: analysis_name,
            words_per_id: words_per_id,
            nodeid_to_index: nodeid_to_index,
            bits_per_id: bits_per_id,
            oper: oper,
            gens: gens,
            action_kills: kills1,
            scope_kills: kills2,
            on_entry: on_entry
        }
    }

    pub fn add_gen(&mut self, id: ast::NodeId, bit: usize) {
        //! Indicates that `id` generates `bit`
        debug!("{} add_gen(id={}, bit={})",
               self.analysis_name, id, bit);
        assert!(self.nodeid_to_index.contains_key(&id));
        assert!(self.bits_per_id > 0);

        let indices = get_cfg_indices(id, &self.nodeid_to_index);
        for &cfgidx in indices {
            let (start, end) = self.compute_id_range(cfgidx);
            let gens = &mut self.gens[start.. end];
            set_bit(gens, bit);
        }
    }

    pub fn add_kill(&mut self, kind: KillFrom, id: ast::NodeId, bit: usize) {
        //! Indicates that `id` kills `bit`
        debug!("{} add_kill(id={}, bit={})",
               self.analysis_name, id, bit);
        assert!(self.nodeid_to_index.contains_key(&id));
        assert!(self.bits_per_id > 0);

        let indices = get_cfg_indices(id, &self.nodeid_to_index);
        for &cfgidx in indices {
            let (start, end) = self.compute_id_range(cfgidx);
            let kills = match kind {
                KillFrom::Execution => &mut self.action_kills[start.. end],
                KillFrom::ScopeEnd =>  &mut self.scope_kills[start.. end],
            };
            set_bit(kills, bit);
        }
    }

    fn apply_gen_kill(&self, cfgidx: CFGIndex, bits: &mut [usize]) {
        //! Applies the gen and kill sets for `cfgidx` to `bits`
        debug!("{} apply_gen_kill(cfgidx={:?}, bits={}) [before]",
               self.analysis_name, cfgidx, mut_bits_to_string(bits));
        assert!(self.bits_per_id > 0);

        let (start, end) = self.compute_id_range(cfgidx);
        let gens = &self.gens[start.. end];
        bitwise(bits, gens, &Union);
        let kills = &self.action_kills[start.. end];
        bitwise(bits, kills, &Subtract);
        let kills = &self.scope_kills[start.. end];
        bitwise(bits, kills, &Subtract);

        debug!("{} apply_gen_kill(cfgidx={:?}, bits={}) [after]",
               self.analysis_name, cfgidx, mut_bits_to_string(bits));
    }

    fn compute_id_range(&self, cfgidx: CFGIndex) -> (usize, usize) {
        let n = cfgidx.node_id();
        let start = n * self.words_per_id;
        let end = start + self.words_per_id;

        assert!(start < self.gens.len());
        assert!(end <= self.gens.len());
        assert!(self.gens.len() == self.action_kills.len());
        assert!(self.gens.len() == self.scope_kills.len());
        assert!(self.gens.len() == self.on_entry.len());

        (start, end)
    }


    pub fn each_bit_on_entry<F>(&self, id: ast::NodeId, mut f: F) -> bool where
        F: FnMut(usize) -> bool,
    {
        //! Iterates through each bit that is set on entry to `id`.
        //! Only useful after `propagate()` has been called.
        if !self.has_bitset_for_nodeid(id) {
            return true;
        }
        let indices = get_cfg_indices(id, &self.nodeid_to_index);
        for &cfgidx in indices {
            if !self.each_bit_for_node(EntryOrExit::Entry, cfgidx, |i| f(i)) {
                return false;
            }
        }
        return true;
    }

    pub fn each_bit_for_node<F>(&self, e: EntryOrExit, cfgidx: CFGIndex, f: F) -> bool where
        F: FnMut(usize) -> bool,
    {
        //! Iterates through each bit that is set on entry/exit to `cfgidx`.
        //! Only useful after `propagate()` has been called.

        if self.bits_per_id == 0 {
            // Skip the surprisingly common degenerate case.  (Note
            // compute_id_range requires self.words_per_id > 0.)
            return true;
        }

        let (start, end) = self.compute_id_range(cfgidx);
        let on_entry = &self.on_entry[start.. end];
        let temp_bits;
        let slice = match e {
            EntryOrExit::Entry => on_entry,
            EntryOrExit::Exit => {
                let mut t = on_entry.to_vec();
                self.apply_gen_kill(cfgidx, &mut t);
                temp_bits = t;
                &temp_bits[..]
            }
        };
        debug!("{} each_bit_for_node({:?}, cfgidx={:?}) bits={}",
               self.analysis_name, e, cfgidx, bits_to_string(slice));
        self.each_bit(slice, f)
    }

    pub fn each_gen_bit<F>(&self, id: ast::NodeId, mut f: F) -> bool where
        F: FnMut(usize) -> bool,
    {
        //! Iterates through each bit in the gen set for `id`.
        if !self.has_bitset_for_nodeid(id) {
            return true;
        }

        if self.bits_per_id == 0 {
            // Skip the surprisingly common degenerate case.  (Note
            // compute_id_range requires self.words_per_id > 0.)
            return true;
        }

        let indices = get_cfg_indices(id, &self.nodeid_to_index);
        for &cfgidx in indices {
            let (start, end) = self.compute_id_range(cfgidx);
            let gens = &self.gens[start.. end];
            debug!("{} each_gen_bit(id={}, gens={})",
                   self.analysis_name, id, bits_to_string(gens));
            if !self.each_bit(gens, |i| f(i)) {
                return false;
            }
        }
        return true;
    }

    fn each_bit<F>(&self, words: &[usize], mut f: F) -> bool where
        F: FnMut(usize) -> bool,
    {
        //! Helper for iterating over the bits in a bit set.
        //! Returns false on the first call to `f` that returns false;
        //! if all calls to `f` return true, then returns true.

        let usize_bits = mem::size_of::<usize>() * 8;
        for (word_index, &word) in words.iter().enumerate() {
            if word != 0 {
                let base_index = word_index * usize_bits;
                for offset in 0..usize_bits {
                    let bit = 1 << offset;
                    if (word & bit) != 0 {
                        // NB: we round up the total number of bits
                        // that we store in any given bit set so that
                        // it is an even multiple of usize::BITS.  This
                        // means that there may be some stray bits at
                        // the end that do not correspond to any
                        // actual value.  So before we callback, check
                        // whether the bit_index is greater than the
                        // actual value the user specified and stop
                        // iterating if so.
                        let bit_index = base_index + offset as usize;
                        if bit_index >= self.bits_per_id {
                            return true;
                        } else if !f(bit_index) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    pub fn add_kills_from_flow_exits(&mut self, cfg: &cfg::CFG) {
        //! Whenever you have a `break` or `continue` statement, flow
        //! exits through any number of enclosing scopes on its way to
        //! the new destination. This function infers the kill bits of
        //! those control operators based on the kill bits associated
        //! with those scopes.
        //!
        //! This is usually called (if it is called at all), after
        //! all add_gen and add_kill calls, but before propagate.

        debug!("{} add_kills_from_flow_exits", self.analysis_name);
        if self.bits_per_id == 0 {
            // Skip the surprisingly common degenerate case.  (Note
            // compute_id_range requires self.words_per_id > 0.)
            return;
        }
        cfg.graph.each_edge(|_edge_index, edge| {
            let flow_exit = edge.source();
            let (start, end) = self.compute_id_range(flow_exit);
            let mut orig_kills = self.scope_kills[start.. end].to_vec();

            let mut changed = false;
            for &node_id in &edge.data.exiting_scopes {
                let opt_cfg_idx = self.nodeid_to_index.get(&node_id);
                match opt_cfg_idx {
                    Some(indices) => {
                        for &cfg_idx in indices {
                            let (start, end) = self.compute_id_range(cfg_idx);
                            let kills = &self.scope_kills[start.. end];
                            if bitwise(&mut orig_kills, kills, &Union) {
                                debug!("scope exits: scope id={} \
                                        (node={:?} of {:?}) added killset: {}",
                                       node_id, cfg_idx, indices,
                                       bits_to_string(kills));
                                changed = true;
                            }
                        }
                    }
                    None => {
                        debug!("{} add_kills_from_flow_exits flow_exit={:?} \
                                no cfg_idx for exiting_scope={}",
                               self.analysis_name, flow_exit, node_id);
                    }
                }
            }

            if changed {
                let bits = &mut self.scope_kills[start.. end];
                debug!("{} add_kills_from_flow_exits flow_exit={:?} bits={} [before]",
                       self.analysis_name, flow_exit, mut_bits_to_string(bits));
                bits.copy_from_slice(&orig_kills[..]);
                debug!("{} add_kills_from_flow_exits flow_exit={:?} bits={} [after]",
                       self.analysis_name, flow_exit, mut_bits_to_string(bits));
            }
            true
        });
    }
}

impl<'a, 'tcx, O:DataFlowOperator+Clone+'static> DataFlowContext<'a, 'tcx, O> {
//                                ^^^^^^^^^^^^^ only needed for pretty printing
    pub fn propagate(&mut self, cfg: &cfg::CFG, body: &hir::Body) {
        //! Performs the data flow analysis.

        if self.bits_per_id == 0 {
            // Optimize the surprisingly common degenerate case.
            return;
        }

        {
            let words_per_id = self.words_per_id;
            let mut propcx = PropagationContext {
                dfcx: &mut *self,
                changed: true
            };

            let mut temp = vec![0; words_per_id];
            while propcx.changed {
                propcx.changed = false;
                propcx.reset(&mut temp);
                propcx.walk_cfg(cfg, &mut temp);
            }
        }

        debug!("Dataflow result for {}:", self.analysis_name);
        debug!("{}", pprust::to_string(self, |s| {
            s.cbox(pprust::indent_unit)?;
            s.ibox(0)?;
            s.print_expr(&body.value)
        }));
    }
}

impl<'a, 'b, 'tcx, O:DataFlowOperator> PropagationContext<'a, 'b, 'tcx, O> {
    fn walk_cfg(&mut self,
                cfg: &cfg::CFG,
                in_out: &mut [usize]) {
        debug!("DataFlowContext::walk_cfg(in_out={}) {}",
               bits_to_string(in_out), self.dfcx.analysis_name);
        assert!(self.dfcx.bits_per_id > 0);

        cfg.graph.each_node(|node_index, node| {
            debug!("DataFlowContext::walk_cfg idx={:?} id={} begin in_out={}",
                   node_index, node.data.id(), bits_to_string(in_out));

            let (start, end) = self.dfcx.compute_id_range(node_index);

            // Initialize local bitvector with state on-entry.
            in_out.copy_from_slice(&self.dfcx.on_entry[start.. end]);

            // Compute state on-exit by applying transfer function to
            // state on-entry.
            self.dfcx.apply_gen_kill(node_index, in_out);

            // Propagate state on-exit from node into its successors.
            self.propagate_bits_into_graph_successors_of(in_out, cfg, node_index);
            true // continue to next node
        });
    }

    fn reset(&mut self, bits: &mut [usize]) {
        let e = if self.dfcx.oper.initial_value() {usize::MAX} else {0};
        for b in bits {
            *b = e;
        }
    }

    fn propagate_bits_into_graph_successors_of(&mut self,
                                               pred_bits: &[usize],
                                               cfg: &cfg::CFG,
                                               cfgidx: CFGIndex) {
        for (_, edge) in cfg.graph.outgoing_edges(cfgidx) {
            self.propagate_bits_into_entry_set_for(pred_bits, edge);
        }
    }

    fn propagate_bits_into_entry_set_for(&mut self,
                                         pred_bits: &[usize],
                                         edge: &cfg::CFGEdge) {
        let source = edge.source();
        let cfgidx = edge.target();
        debug!("{} propagate_bits_into_entry_set_for(pred_bits={}, {:?} to {:?})",
               self.dfcx.analysis_name, bits_to_string(pred_bits), source, cfgidx);
        assert!(self.dfcx.bits_per_id > 0);

        let (start, end) = self.dfcx.compute_id_range(cfgidx);
        let changed = {
            // (scoping mutable borrow of self.dfcx.on_entry)
            let on_entry = &mut self.dfcx.on_entry[start.. end];
            bitwise(on_entry, pred_bits, &self.dfcx.oper)
        };
        if changed {
            debug!("{} changed entry set for {:?} to {}",
                   self.dfcx.analysis_name, cfgidx,
                   bits_to_string(&self.dfcx.on_entry[start.. end]));
            self.changed = true;
        }
    }
}

fn mut_bits_to_string(words: &mut [usize]) -> String {
    bits_to_string(words)
}

fn bits_to_string(words: &[usize]) -> String {
    let mut result = String::new();
    let mut sep = '[';

    // Note: this is a little endian printout of bytes.

    for &word in words {
        let mut v = word;
        for _ in 0..mem::size_of::<usize>() {
            result.push(sep);
            result.push_str(&format!("{:02x}", v & 0xFF));
            v >>= 8;
            sep = '-';
        }
    }
    result.push(']');
    return result
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

fn set_bit(words: &mut [usize], bit: usize) -> bool {
    debug!("set_bit: words={} bit={}",
           mut_bits_to_string(words), bit_str(bit));
    let usize_bits = mem::size_of::<usize>() * 8;
    let word = bit / usize_bits;
    let bit_in_word = bit % usize_bits;
    let bit_mask = 1 << bit_in_word;
    debug!("word={} bit_in_word={} bit_mask={}", word, bit_in_word, bit_mask);
    let oldv = words[word];
    let newv = oldv | bit_mask;
    words[word] = newv;
    oldv != newv
}

fn bit_str(bit: usize) -> String {
    let byte = bit >> 3;
    let lobits = 1 << (bit & 0b111);
    format!("[{}:{}-{:02x}]", bit, byte, lobits)
}

struct Union;
impl BitwiseOperator for Union {
    fn join(&self, a: usize, b: usize) -> usize { a | b }
}
struct Subtract;
impl BitwiseOperator for Subtract {
    fn join(&self, a: usize, b: usize) -> usize { a & !b }
}
