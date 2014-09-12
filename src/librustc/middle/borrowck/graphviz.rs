// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides linkage between rustc::middle::graph and
//! libgraphviz traits, specialized to attaching borrowck analysis
//! data to rendered labels.

/// For clarity, rename the graphviz crate locally to dot.
use graphviz as dot;
pub use middle::cfg::graphviz::{Node, Edge};
use middle::cfg::graphviz as cfg_dot;

use middle::borrowck;
use middle::borrowck::{BorrowckCtxt, LoanPath};
use middle::cfg::{CFGIndex};
use middle::dataflow::{DataFlowOperator, DataFlowContext, EntryOrExit};
use middle::dataflow;

use std::rc::Rc;
use std::str;

#[deriving(Show)]
pub enum Variant {
    Loans,
    Moves,
    Assigns,
}

impl Variant {
    pub fn short_name(&self) -> &'static str {
        match *self {
            Loans   => "loans",
            Moves   => "moves",
            Assigns => "assigns",
        }
    }
}

pub struct DataflowLabeller<'a, 'tcx: 'a> {
    pub inner: cfg_dot::LabelledCFG<'a, 'tcx>,
    pub variants: Vec<Variant>,
    pub borrowck_ctxt: &'a BorrowckCtxt<'a, 'tcx>,
    pub analysis_data: &'a borrowck::AnalysisData<'a, 'tcx>,
}

impl<'a, 'tcx> DataflowLabeller<'a, 'tcx> {
    fn dataflow_for(&self, e: EntryOrExit, n: &Node<'a>) -> String {
        let id = n.val1().data.id;
        debug!("dataflow_for({}, id={}) {}", e, id, self.variants);
        let mut sets = "".to_string();
        let mut seen_one = false;
        for &variant in self.variants.iter() {
            if seen_one { sets.push_str(" "); } else { seen_one = true; }
            sets.push_str(variant.short_name());
            sets.push_str(": ");
            sets.push_str(self.dataflow_for_variant(e, n, variant).as_slice());
        }
        sets
    }

    fn dataflow_for_variant(&self, e: EntryOrExit, n: &Node, v: Variant) -> String {
        let cfgidx = n.val0();
        match v {
            Loans   => self.dataflow_loans_for(e, cfgidx),
            Moves   => self.dataflow_moves_for(e, cfgidx),
            Assigns => self.dataflow_assigns_for(e, cfgidx),
        }
    }

    fn build_set<O:DataFlowOperator>(&self,
                                     e: EntryOrExit,
                                     cfgidx: CFGIndex,
                                     dfcx: &DataFlowContext<'a, 'tcx, O>,
                                     to_lp: |uint| -> Rc<LoanPath>) -> String {
        let mut saw_some = false;
        let mut set = "{".to_string();
        dfcx.each_bit_for_node(e, cfgidx, |index| {
            let lp = to_lp(index);
            if saw_some {
                set.push_str(", ");
            }
            let loan_str = self.borrowck_ctxt.loan_path_to_string(&*lp);
            set.push_str(loan_str.as_slice());
            saw_some = true;
            true
        });
        set.append("}")
    }

    fn dataflow_loans_for(&self, e: EntryOrExit, cfgidx: CFGIndex) -> String {
        let dfcx = &self.analysis_data.loans;
        let loan_index_to_path = |loan_index| {
            let all_loans = &self.analysis_data.all_loans;
            all_loans.get(loan_index).loan_path()
        };
        self.build_set(e, cfgidx, dfcx, loan_index_to_path)
    }

    fn dataflow_moves_for(&self, e: EntryOrExit, cfgidx: CFGIndex) -> String {
        let dfcx = &self.analysis_data.move_data.dfcx_moves;
        let move_index_to_path = |move_index| {
            let move_data = &self.analysis_data.move_data.move_data;
            let moves = move_data.moves.borrow();
            let move = moves.get(move_index);
            move_data.path_loan_path(move.path)
        };
        self.build_set(e, cfgidx, dfcx, move_index_to_path)
    }

    fn dataflow_assigns_for(&self, e: EntryOrExit, cfgidx: CFGIndex) -> String {
        let dfcx = &self.analysis_data.move_data.dfcx_assign;
        let assign_index_to_path = |assign_index| {
            let move_data = &self.analysis_data.move_data.move_data;
            let assignments = move_data.var_assignments.borrow();
            let assignment = assignments.get(assign_index);
            move_data.path_loan_path(assignment.path)
        };
        self.build_set(e, cfgidx, dfcx, assign_index_to_path)
    }
}

impl<'a, 'tcx> dot::Labeller<'a, Node<'a>, Edge<'a>> for DataflowLabeller<'a, 'tcx> {
    fn graph_id(&'a self) -> dot::Id<'a> { self.inner.graph_id() }
    fn node_id(&'a self, n: &Node<'a>) -> dot::Id<'a> { self.inner.node_id(n) }
    fn node_label(&'a self, n: &Node<'a>) -> dot::LabelText<'a> {
        let prefix = self.dataflow_for(dataflow::Entry, n);
        let suffix = self.dataflow_for(dataflow::Exit, n);
        let inner_label = self.inner.node_label(n);
        inner_label
            .prefix_line(dot::LabelStr(str::Owned(prefix)))
            .suffix_line(dot::LabelStr(str::Owned(suffix)))
    }
    fn edge_label(&'a self, e: &Edge<'a>) -> dot::LabelText<'a> { self.inner.edge_label(e) }
}

impl<'a, 'tcx> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for DataflowLabeller<'a, 'tcx> {
    fn nodes(&self) -> dot::Nodes<'a, Node<'a>> { self.inner.nodes() }
    fn edges(&self) -> dot::Edges<'a, Edge<'a>> { self.inner.edges() }
    fn source(&self, edge: &Edge<'a>) -> Node<'a> { self.inner.source(edge) }
    fn target(&self, edge: &Edge<'a>) -> Node<'a> { self.inner.target(edge) }
}
