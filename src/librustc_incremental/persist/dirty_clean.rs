// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Debugging code to test the state of the dependency graph just
//! after it is loaded from disk. For each node marked with
//! `#[rustc_clean]` or `#[rustc_dirty]`, we will check that a
//! suitable node for that item either appears or does not appear in
//! the dep-graph, as appropriate:
//!
//! - `#[rustc_dirty(label="TypeckItemBody", cfg="rev2")]` if we are
//!   in `#[cfg(rev2)]`, then there MUST NOT be a node
//!   `DepNode::TypeckItemBody(X)` where `X` is the def-id of the
//!   current node.
//! - `#[rustc_clean(label="TypeckItemBody", cfg="rev2")]` same as above,
//!   except that the node MUST exist.
//!
//! Errors are reported if we are in the suitable configuration but
//! the required condition is not met.

use rustc::dep_graph::{DepGraphQuery, DepNode};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::Visitor;
use syntax::ast::{self, Attribute, MetaItem};
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::InternedString;
use rustc::ty;

const DIRTY: &'static str = "rustc_dirty";
const CLEAN: &'static str = "rustc_clean";
const LABEL: &'static str = "label";
const CFG: &'static str = "cfg";

pub fn check_dirty_clean_annotations(tcx: &ty::TyCtxt) {
    let _ignore = tcx.dep_graph.in_ignore();
    let query = tcx.dep_graph.query();
    let krate = tcx.map.krate();
    krate.visit_all_items(&mut DirtyCleanVisitor {
        tcx: tcx,
        query: &query,
    });
}

pub struct DirtyCleanVisitor<'a, 'tcx:'a> {
    tcx: &'a ty::TyCtxt<'tcx>,
    query: &'a DepGraphQuery<DefId>,
}

impl<'a, 'tcx> DirtyCleanVisitor<'a, 'tcx> {
    fn expect_associated_value(&self, item: &MetaItem) -> InternedString {
        if let Some(value) = item.value_str() {
            value
        } else {
            self.tcx.sess.span_fatal(
                item.span,
                &format!("associated value expected for `{}`", item.name()));
        }
    }

    /// Given a `#[rustc_dirty]` or `#[rustc_clean]` attribute, scan
    /// for a `cfg="foo"` attribute and check whether we have a cfg
    /// flag called `foo`.
    fn check_config(&self, attr: &ast::Attribute) -> bool {
        debug!("check_config(attr={:?})", attr);
        let config = &self.tcx.map.krate().config;
        debug!("check_config: config={:?}", config);
        for item in attr.meta_item_list().unwrap_or(&[]) {
            if item.check_name(CFG) {
                let value = self.expect_associated_value(item);
                debug!("check_config: searching for cfg {:?}", value);
                for cfg in &config[..] {
                    if cfg.check_name(&value[..]) {
                        debug!("check_config: matched {:?}", cfg);
                        return true;
                    }
                }
            }
        }
        debug!("check_config: no match found");
        return false;
    }

    fn dep_node(&self, attr: &Attribute, def_id: DefId) -> DepNode<DefId> {
        for item in attr.meta_item_list().unwrap_or(&[]) {
            if item.check_name(LABEL) {
                let value = self.expect_associated_value(item);
                match DepNode::from_label_string(&value[..], def_id) {
                    Ok(def_id) => return def_id,
                    Err(()) => {
                        self.tcx.sess.span_fatal(
                            item.span,
                            &format!("dep-node label `{}` not recognized", value));
                    }
                }
            }
        }

        self.tcx.sess.span_fatal(attr.span, "no `label` found");
    }

    fn dep_node_str(&self, dep_node: DepNode<DefId>) -> DepNode<String> {
        dep_node.map_def(|&def_id| Some(self.tcx.item_path_str(def_id))).unwrap()
    }

    fn assert_dirty(&self, item: &hir::Item, dep_node: DepNode<DefId>) {
        debug!("assert_dirty({:?})", dep_node);

        if self.query.contains_node(&dep_node) {
            let dep_node_str = self.dep_node_str(dep_node);
            self.tcx.sess.span_err(
                item.span,
                &format!("`{:?}` found in dep graph, but should be dirty", dep_node_str));
        }
    }

    fn assert_clean(&self, item: &hir::Item, dep_node: DepNode<DefId>) {
        debug!("assert_clean({:?})", dep_node);

        if !self.query.contains_node(&dep_node) {
            let dep_node_str = self.dep_node_str(dep_node);
            self.tcx.sess.span_err(
                item.span,
                &format!("`{:?}` not found in dep graph, but should be clean", dep_node_str));
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for DirtyCleanVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_id = self.tcx.map.local_def_id(item.id);
        for attr in self.tcx.get_attrs(def_id).iter() {
            if attr.check_name(DIRTY) {
                if self.check_config(attr) {
                    self.assert_dirty(item, self.dep_node(attr, def_id));
                }
            } else if attr.check_name(CLEAN) {
                if self.check_config(attr) {
                    self.assert_clean(item, self.dep_node(attr, def_id));
                }
            }
        }
    }
}

