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
//! after it is loaded from disk and just after it has been saved.
//! For each node marked with `#[rustc_clean]` or `#[rustc_dirty]`,
//! we will check that a suitable node for that item either appears
//! or does not appear in the dep-graph, as appropriate:
//!
//! - `#[rustc_dirty(label="TypeckTables", cfg="rev2")]` if we are
//!   in `#[cfg(rev2)]`, then there MUST NOT be a node
//!   `DepNode::TypeckTables(X)` where `X` is the def-id of the
//!   current node.
//! - `#[rustc_clean(label="TypeckTables", cfg="rev2")]` same as above,
//!   except that the node MUST exist.
//!
//! Errors are reported if we are in the suitable configuration but
//! the required condition is not met.
//!
//! The `#[rustc_metadata_dirty]` and `#[rustc_metadata_clean]` attributes
//! can be used to check the incremental compilation hash (ICH) values of
//! metadata exported in rlibs.
//!
//! - If a node is marked with `#[rustc_metadata_clean(cfg="rev2")]` we
//!   check that the metadata hash for that node is the same for "rev2"
//!   it was for "rev1".
//! - If a node is marked with `#[rustc_metadata_dirty(cfg="rev2")]` we
//!   check that the metadata hash for that node is *different* for "rev2"
//!   than it was for "rev1".
//!
//! Note that the metadata-testing attributes must never specify the
//! first revision. This would lead to a crash since there is no
//! previous revision to compare things to.
//!

use super::directory::RetracedDefIdDirectory;
use super::load::DirtyNodes;
use rustc::dep_graph::{DepGraphQuery, DepNode};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::intravisit;
use syntax::ast::{self, Attribute, NestedMetaItem};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use syntax_pos::Span;
use rustc::ty::TyCtxt;
use ich::Fingerprint;

use {ATTR_DIRTY, ATTR_CLEAN, ATTR_DIRTY_METADATA, ATTR_CLEAN_METADATA};

const LABEL: &'static str = "label";
const CFG: &'static str = "cfg";

pub fn check_dirty_clean_annotations<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                               dirty_inputs: &DirtyNodes,
                                               retraced: &RetracedDefIdDirectory) {
    // can't add `#[rustc_dirty]` etc without opting in to this feature
    if !tcx.sess.features.borrow().rustc_attrs {
        return;
    }

    let _ignore = tcx.dep_graph.in_ignore();
    let dirty_inputs: FxHashSet<DepNode<DefId>> =
        dirty_inputs.keys()
                    .filter_map(|d| retraced.map(d))
                    .collect();
    let query = tcx.dep_graph.query();
    debug!("query-nodes: {:?}", query.nodes());
    let krate = tcx.hir.krate();
    let mut dirty_clean_visitor = DirtyCleanVisitor {
        tcx: tcx,
        query: &query,
        dirty_inputs: dirty_inputs,
        checked_attrs: FxHashSet(),
    };
    krate.visit_all_item_likes(&mut dirty_clean_visitor);

    let mut all_attrs = FindAllAttrs {
        tcx: tcx,
        attr_names: vec![ATTR_DIRTY, ATTR_CLEAN],
        found_attrs: vec![],
    };
    intravisit::walk_crate(&mut all_attrs, krate);

    // Note that we cannot use the existing "unused attribute"-infrastructure
    // here, since that is running before trans. This is also the reason why
    // all trans-specific attributes are `Whitelisted` in syntax::feature_gate.
    all_attrs.report_unchecked_attrs(&dirty_clean_visitor.checked_attrs);
}

pub struct DirtyCleanVisitor<'a, 'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    query: &'a DepGraphQuery<DefId>,
    dirty_inputs: FxHashSet<DepNode<DefId>>,
    checked_attrs: FxHashSet<ast::AttrId>,
}

impl<'a, 'tcx> DirtyCleanVisitor<'a, 'tcx> {
    fn dep_node(&self, attr: &Attribute, def_id: DefId) -> DepNode<DefId> {
        for item in attr.meta_item_list().unwrap_or(&[]) {
            if item.check_name(LABEL) {
                let value = expect_associated_value(self.tcx, item);
                match DepNode::from_label_string(&value.as_str(), def_id) {
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

    fn dep_node_str(&self, dep_node: &DepNode<DefId>) -> DepNode<String> {
        dep_node.map_def(|&def_id| Some(self.tcx.item_path_str(def_id))).unwrap()
    }

    fn assert_dirty(&self, item_span: Span, dep_node: DepNode<DefId>) {
        debug!("assert_dirty({:?})", dep_node);

        match dep_node {
            DepNode::Krate |
            DepNode::Hir(_) |
            DepNode::HirBody(_) => {
                // HIR nodes are inputs, so if we are asserting that the HIR node is
                // dirty, we check the dirty input set.
                if !self.dirty_inputs.contains(&dep_node) {
                    let dep_node_str = self.dep_node_str(&dep_node);
                    self.tcx.sess.span_err(
                        item_span,
                        &format!("`{:?}` not found in dirty set, but should be dirty",
                                 dep_node_str));
                }
            }
            _ => {
                // Other kinds of nodes would be targets, so check if
                // the dep-graph contains the node.
                if self.query.contains_node(&dep_node) {
                    let dep_node_str = self.dep_node_str(&dep_node);
                    self.tcx.sess.span_err(
                        item_span,
                        &format!("`{:?}` found in dep graph, but should be dirty", dep_node_str));
                }
            }
        }
    }

    fn assert_clean(&self, item_span: Span, dep_node: DepNode<DefId>) {
        debug!("assert_clean({:?})", dep_node);

        match dep_node {
            DepNode::Krate |
            DepNode::Hir(_) |
            DepNode::HirBody(_) => {
                // For HIR nodes, check the inputs.
                if self.dirty_inputs.contains(&dep_node) {
                    let dep_node_str = self.dep_node_str(&dep_node);
                    self.tcx.sess.span_err(
                        item_span,
                        &format!("`{:?}` found in dirty-node set, but should be clean",
                                 dep_node_str));
                }
            }
            _ => {
                // Otherwise, check if the dep-node exists.
                if !self.query.contains_node(&dep_node) {
                    let dep_node_str = self.dep_node_str(&dep_node);
                    self.tcx.sess.span_err(
                        item_span,
                        &format!("`{:?}` not found in dep graph, but should be clean",
                                 dep_node_str));
                }
            }
        }
    }

    fn check_item(&mut self, item_id: ast::NodeId, item_span: Span) {
        let def_id = self.tcx.hir.local_def_id(item_id);
        for attr in self.tcx.get_attrs(def_id).iter() {
            if attr.check_name(ATTR_DIRTY) {
                if check_config(self.tcx, attr) {
                    self.checked_attrs.insert(attr.id);
                    self.assert_dirty(item_span, self.dep_node(attr, def_id));
                }
            } else if attr.check_name(ATTR_CLEAN) {
                if check_config(self.tcx, attr) {
                    self.checked_attrs.insert(attr.id);
                    self.assert_clean(item_span, self.dep_node(attr, def_id));
                }
            }
        }
    }
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for DirtyCleanVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.check_item(item.id, item.span);
    }

    fn visit_trait_item(&mut self, item: &hir::TraitItem) {
        self.check_item(item.id, item.span);
    }

    fn visit_impl_item(&mut self, item: &hir::ImplItem) {
        self.check_item(item.id, item.span);
    }
}

pub fn check_dirty_clean_metadata<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  prev_metadata_hashes: &FxHashMap<DefId, Fingerprint>,
                                  current_metadata_hashes: &FxHashMap<DefId, Fingerprint>) {
    if !tcx.sess.opts.debugging_opts.query_dep_graph {
        return;
    }

    tcx.dep_graph.with_ignore(||{
        let krate = tcx.hir.krate();
        let mut dirty_clean_visitor = DirtyCleanMetadataVisitor {
            tcx: tcx,
            prev_metadata_hashes: prev_metadata_hashes,
            current_metadata_hashes: current_metadata_hashes,
            checked_attrs: FxHashSet(),
        };
        krate.visit_all_item_likes(&mut dirty_clean_visitor);

        let mut all_attrs = FindAllAttrs {
            tcx: tcx,
            attr_names: vec![ATTR_DIRTY_METADATA, ATTR_CLEAN_METADATA],
            found_attrs: vec![],
        };
        intravisit::walk_crate(&mut all_attrs, krate);

        // Note that we cannot use the existing "unused attribute"-infrastructure
        // here, since that is running before trans. This is also the reason why
        // all trans-specific attributes are `Whitelisted` in syntax::feature_gate.
        all_attrs.report_unchecked_attrs(&dirty_clean_visitor.checked_attrs);
    });
}

pub struct DirtyCleanMetadataVisitor<'a, 'tcx:'a, 'm> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    prev_metadata_hashes: &'m FxHashMap<DefId, Fingerprint>,
    current_metadata_hashes: &'m FxHashMap<DefId, Fingerprint>,
    checked_attrs: FxHashSet<ast::AttrId>,
}

impl<'a, 'tcx, 'm> ItemLikeVisitor<'tcx> for DirtyCleanMetadataVisitor<'a, 'tcx, 'm> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.check_item(item.id, item.span);

        if let hir::ItemEnum(ref def, _) = item.node {
            for v in &def.variants {
                self.check_item(v.node.data.id(), v.span);
            }
        }
    }

    fn visit_trait_item(&mut self, item: &hir::TraitItem) {
        self.check_item(item.id, item.span);
    }

    fn visit_impl_item(&mut self, item: &hir::ImplItem) {
        self.check_item(item.id, item.span);
    }
}

impl<'a, 'tcx, 'm> DirtyCleanMetadataVisitor<'a, 'tcx, 'm> {

    fn check_item(&mut self, item_id: ast::NodeId, item_span: Span) {
        let def_id = self.tcx.hir.local_def_id(item_id);

        for attr in self.tcx.get_attrs(def_id).iter() {
            if attr.check_name(ATTR_DIRTY_METADATA) {
                if check_config(self.tcx, attr) {
                    self.checked_attrs.insert(attr.id);
                    self.assert_state(false, def_id, item_span);
                }
            } else if attr.check_name(ATTR_CLEAN_METADATA) {
                if check_config(self.tcx, attr) {
                    self.checked_attrs.insert(attr.id);
                    self.assert_state(true, def_id, item_span);
                }
            }
        }
    }

    fn assert_state(&self, should_be_clean: bool, def_id: DefId, span: Span) {
        let item_path = self.tcx.item_path_str(def_id);
        debug!("assert_state({})", item_path);

        if let Some(&prev_hash) = self.prev_metadata_hashes.get(&def_id) {
            let hashes_are_equal = prev_hash == self.current_metadata_hashes[&def_id];

            if should_be_clean && !hashes_are_equal {
                self.tcx.sess.span_err(
                        span,
                        &format!("Metadata hash of `{}` is dirty, but should be clean",
                                 item_path));
            }

            let should_be_dirty = !should_be_clean;
            if should_be_dirty && hashes_are_equal {
                self.tcx.sess.span_err(
                        span,
                        &format!("Metadata hash of `{}` is clean, but should be dirty",
                                 item_path));
            }
        } else {
            self.tcx.sess.span_err(
                        span,
                        &format!("Could not find previous metadata hash of `{}`",
                                 item_path));
        }
    }
}

/// Given a `#[rustc_dirty]` or `#[rustc_clean]` attribute, scan
/// for a `cfg="foo"` attribute and check whether we have a cfg
/// flag called `foo`.
fn check_config(tcx: TyCtxt, attr: &Attribute) -> bool {
    debug!("check_config(attr={:?})", attr);
    let config = &tcx.sess.parse_sess.config;
    debug!("check_config: config={:?}", config);
    for item in attr.meta_item_list().unwrap_or(&[]) {
        if item.check_name(CFG) {
            let value = expect_associated_value(tcx, item);
            debug!("check_config: searching for cfg {:?}", value);
            return config.contains(&(value, None));
        }
    }

    tcx.sess.span_fatal(
        attr.span,
        &format!("no cfg attribute"));
}

fn expect_associated_value(tcx: TyCtxt, item: &NestedMetaItem) -> ast::Name {
    if let Some(value) = item.value_str() {
        value
    } else {
        let msg = if let Some(name) = item.name() {
            format!("associated value expected for `{}`", name)
        } else {
            "expected an associated value".to_string()
        };

        tcx.sess.span_fatal(item.span, &msg);
    }
}


// A visitor that collects all #[rustc_dirty]/#[rustc_clean] attributes from
// the HIR. It is used to verfiy that we really ran checks for all annotated
// nodes.
pub struct FindAllAttrs<'a, 'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    attr_names: Vec<&'static str>,
    found_attrs: Vec<&'tcx Attribute>,
}

impl<'a, 'tcx> FindAllAttrs<'a, 'tcx> {

    fn is_active_attr(&mut self, attr: &Attribute) -> bool {
        for attr_name in &self.attr_names {
            if attr.check_name(attr_name) && check_config(self.tcx, attr) {
                return true;
            }
        }

        false
    }

    fn report_unchecked_attrs(&self, checked_attrs: &FxHashSet<ast::AttrId>) {
        for attr in &self.found_attrs {
            if !checked_attrs.contains(&attr.id) {
                self.tcx.sess.span_err(attr.span, &format!("found unchecked \
                    #[rustc_dirty]/#[rustc_clean] attribute"));
            }
        }
    }
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for FindAllAttrs<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
        intravisit::NestedVisitorMap::All(&self.tcx.hir)
    }

    fn visit_attribute(&mut self, attr: &'tcx Attribute) {
        if self.is_active_attr(attr) {
            self.found_attrs.push(attr);
        }
    }
}
