// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use dep_graph::DepNode;
use hir::map as hir_map;
use lint;
use hir::def::Def;
use hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId, DefIndex, LOCAL_CRATE};
use ty::{self, TyCtxt};
use middle::privacy::AccessLevels;
use syntax::symbol::Symbol;
use syntax_pos::{Span, DUMMY_SP};
use syntax::ast;
use syntax::ast::{NodeId, Attribute};
use syntax::feature_gate::{GateIssue, emit_feature_err, find_lang_feature_accepted_version};
use syntax::attr::{self, Stability, Deprecation};
use util::nodemap::{DefIdMap, FxHashSet, FxHashMap};

use hir;
use hir::{Item, Generics, StructField, Variant};
use hir::intravisit::{self, Visitor, NestedVisitorMap};

use std::mem::replace;
use std::cmp::Ordering;

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Copy, Debug, Eq, Hash)]
pub enum StabilityLevel {
    Unstable,
    Stable,
}

impl StabilityLevel {
    pub fn from_attr_level(level: &attr::StabilityLevel) -> Self {
        if level.is_stable() { Stable } else { Unstable }
    }
}

#[derive(PartialEq)]
enum AnnotationKind {
    // Annotation is required if not inherited from unstable parents
    Required,
    // Annotation is useless, reject it
    Prohibited,
    // Annotation itself is useless, but it can be propagated to children
    Container,
}

/// An entry in the `depr_map`.
#[derive(Clone)]
pub struct DeprecationEntry {
    /// The metadata of the attribute associated with this entry.
    pub attr: Deprecation,
    /// The def id where the attr was originally attached. `None` for non-local
    /// `DefId`'s.
    origin: Option<DefIndex>,
}

impl DeprecationEntry {
    fn local(attr: Deprecation, id: DefId) -> DeprecationEntry {
        assert!(id.is_local());
        DeprecationEntry {
            attr: attr,
            origin: Some(id.index),
        }
    }

    fn external(attr: Deprecation) -> DeprecationEntry {
        DeprecationEntry {
            attr: attr,
            origin: None,
        }
    }

    pub fn same_origin(&self, other: &DeprecationEntry) -> bool {
        match (self.origin, other.origin) {
            (Some(o1), Some(o2)) => o1 == o2,
            _ => false
        }
    }
}

/// A stability index, giving the stability level for items and methods.
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    stab_map: DefIdMap<Option<&'tcx Stability>>,
    depr_map: DefIdMap<Option<DeprecationEntry>>,

    /// Maps for each crate whether it is part of the staged API.
    staged_api: FxHashMap<CrateNum, bool>,

    /// Features enabled for this crate.
    active_features: FxHashSet<Symbol>,

    /// Features used by this crate. Updated before and during typeck.
    used_features: FxHashMap<Symbol, attr::StabilityLevel>
}

// A private tree-walker for producing an Index.
struct Annotator<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    index: &'a mut Index<'tcx>,
    parent_stab: Option<&'tcx Stability>,
    parent_depr: Option<DeprecationEntry>,
    in_trait_impl: bool,
}

impl<'a, 'tcx: 'a> Annotator<'a, 'tcx> {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate<F>(&mut self, id: NodeId, attrs: &[Attribute],
                   item_sp: Span, kind: AnnotationKind, visit_children: F)
        where F: FnOnce(&mut Self)
    {
        if self.index.staged_api[&LOCAL_CRATE] && self.tcx.sess.features.borrow().staged_api {
            debug!("annotate(id = {:?}, attrs = {:?})", id, attrs);
            if let Some(..) = attr::find_deprecation(self.tcx.sess.diagnostic(), attrs, item_sp) {
                self.tcx.sess.span_err(item_sp, "`#[deprecated]` cannot be used in staged api, \
                                                 use `#[rustc_deprecated]` instead");
            }
            if let Some(mut stab) = attr::find_stability(self.tcx.sess.diagnostic(),
                                                         attrs, item_sp) {
                // Error if prohibited, or can't inherit anything from a container
                if kind == AnnotationKind::Prohibited ||
                   (kind == AnnotationKind::Container &&
                    stab.level.is_stable() &&
                    stab.rustc_depr.is_none()) {
                    self.tcx.sess.span_err(item_sp, "This stability annotation is useless");
                }

                debug!("annotate: found {:?}", stab);
                // If parent is deprecated and we're not, inherit this by merging
                // deprecated_since and its reason.
                if let Some(parent_stab) = self.parent_stab {
                    if parent_stab.rustc_depr.is_some() && stab.rustc_depr.is_none() {
                        stab.rustc_depr = parent_stab.rustc_depr.clone()
                    }
                }

                let stab = self.tcx.intern_stability(stab);

                // Check if deprecated_since < stable_since. If it is,
                // this is *almost surely* an accident.
                if let (&Some(attr::RustcDeprecation {since: dep_since, ..}),
                        &attr::Stable {since: stab_since}) = (&stab.rustc_depr, &stab.level) {
                    // Explicit version of iter::order::lt to handle parse errors properly
                    for (dep_v, stab_v) in
                            dep_since.as_str().split(".").zip(stab_since.as_str().split(".")) {
                        if let (Ok(dep_v), Ok(stab_v)) = (dep_v.parse::<u64>(), stab_v.parse()) {
                            match dep_v.cmp(&stab_v) {
                                Ordering::Less => {
                                    self.tcx.sess.span_err(item_sp, "An API can't be stabilized \
                                                                     after it is deprecated");
                                    break
                                }
                                Ordering::Equal => continue,
                                Ordering::Greater => break,
                            }
                        } else {
                            // Act like it isn't less because the question is now nonsensical,
                            // and this makes us not do anything else interesting.
                            self.tcx.sess.span_err(item_sp, "Invalid stability or deprecation \
                                                             version found");
                            break
                        }
                    }
                }

                let def_id = self.tcx.map.local_def_id(id);
                self.index.stab_map.insert(def_id, Some(stab));

                let orig_parent_stab = replace(&mut self.parent_stab, Some(stab));
                visit_children(self);
                self.parent_stab = orig_parent_stab;
            } else {
                debug!("annotate: not found, parent = {:?}", self.parent_stab);
                if let Some(stab) = self.parent_stab {
                    if stab.level.is_unstable() {
                        let def_id = self.tcx.map.local_def_id(id);
                        self.index.stab_map.insert(def_id, Some(stab));
                    }
                }
                visit_children(self);
            }
        } else {
            // Emit errors for non-staged-api crates.
            for attr in attrs {
                let tag = attr.name();
                if tag == "unstable" || tag == "stable" || tag == "rustc_deprecated" {
                    attr::mark_used(attr);
                    self.tcx.sess.span_err(attr.span(), "stability attributes may not be used \
                                                         outside of the standard library");
                }
            }

            if let Some(depr) = attr::find_deprecation(self.tcx.sess.diagnostic(), attrs, item_sp) {
                if kind == AnnotationKind::Prohibited {
                    self.tcx.sess.span_err(item_sp, "This deprecation annotation is useless");
                }

                // `Deprecation` is just two pointers, no need to intern it
                let def_id = self.tcx.map.local_def_id(id);
                let depr_entry = Some(DeprecationEntry::local(depr, def_id));
                self.index.depr_map.insert(def_id, depr_entry.clone());

                let orig_parent_depr = replace(&mut self.parent_depr, depr_entry);
                visit_children(self);
                self.parent_depr = orig_parent_depr;
            } else if let parent_depr @ Some(_) = self.parent_depr.clone() {
                let def_id = self.tcx.map.local_def_id(id);
                self.index.depr_map.insert(def_id, parent_depr);
                visit_children(self);
            } else {
                visit_children(self);
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Annotator<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.map)
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        let orig_in_trait_impl = self.in_trait_impl;
        let mut kind = AnnotationKind::Required;
        match i.node {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemImpl(.., None, _, _) | hir::ItemForeignMod(..) => {
                self.in_trait_impl = false;
                kind = AnnotationKind::Container;
            }
            hir::ItemImpl(.., Some(_), _, _) => {
                self.in_trait_impl = true;
            }
            hir::ItemStruct(ref sd, _) => {
                if !sd.is_struct() {
                    self.annotate(sd.id(), &i.attrs, i.span, AnnotationKind::Required, |_| {})
                }
            }
            _ => {}
        }

        self.annotate(i.id, &i.attrs, i.span, kind, |v| {
            intravisit::walk_item(v, i)
        });
        self.in_trait_impl = orig_in_trait_impl;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        self.annotate(ti.id, &ti.attrs, ti.span, AnnotationKind::Required, |v| {
            intravisit::walk_trait_item(v, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let kind = if self.in_trait_impl {
            AnnotationKind::Prohibited
        } else {
            AnnotationKind::Required
        };
        self.annotate(ii.id, &ii.attrs, ii.span, kind, |v| {
            intravisit::walk_impl_item(v, ii);
        });
    }

    fn visit_variant(&mut self, var: &'tcx Variant, g: &'tcx Generics, item_id: NodeId) {
        self.annotate(var.node.data.id(), &var.node.attrs, var.span, AnnotationKind::Required, |v| {
            intravisit::walk_variant(v, var, g, item_id);
        })
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField) {
        self.annotate(s.id, &s.attrs, s.span, AnnotationKind::Required, |v| {
            intravisit::walk_struct_field(v, s);
        });
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem) {
        self.annotate(i.id, &i.attrs, i.span, AnnotationKind::Required, |v| {
            intravisit::walk_foreign_item(v, i);
        });
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
        self.annotate(md.id, &md.attrs, md.span, AnnotationKind::Required, |_| {});
    }
}

struct MissingStabilityAnnotations<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    access_levels: &'a AccessLevels,
}

impl<'a, 'tcx: 'a> MissingStabilityAnnotations<'a, 'tcx> {
    fn check_missing_stability(&self, id: NodeId, span: Span) {
        let def_id = self.tcx.map.local_def_id(id);
        let is_error = !self.tcx.sess.opts.test &&
                        !self.tcx.stability.borrow().stab_map.contains_key(&def_id) &&
                        self.access_levels.is_reachable(id);
        if is_error {
            self.tcx.sess.span_err(span, "This node does not have a stability attribute");
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.map)
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        match i.node {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemImpl(.., None, _, _) | hir::ItemForeignMod(..) => {}

            _ => self.check_missing_stability(i.id, i.span)
        }

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        self.check_missing_stability(ti.id, ti.span);
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let impl_def_id = self.tcx.map.local_def_id(self.tcx.map.get_parent(ii.id));
        if self.tcx.impl_trait_ref(impl_def_id).is_none() {
            self.check_missing_stability(ii.id, ii.span);
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant, g: &'tcx Generics, item_id: NodeId) {
        self.check_missing_stability(var.node.data.id(), var.span);
        intravisit::walk_variant(self, var, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField) {
        self.check_missing_stability(s.id, s.span);
        intravisit::walk_struct_field(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem) {
        self.check_missing_stability(i.id, i.span);
        intravisit::walk_foreign_item(self, i);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
        self.check_missing_stability(md.id, md.span);
    }
}

impl<'a, 'tcx> Index<'tcx> {
    /// Construct the stability index for a crate being compiled.
    pub fn build(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>) {
        let ref active_lib_features = tcx.sess.features.borrow().declared_lib_features;

        // Put the active features into a map for quick lookup
        self.active_features = active_lib_features.iter().map(|&(ref s, _)| s.clone()).collect();

        let _task = tcx.dep_graph.in_task(DepNode::StabilityIndex);
        let krate = tcx.map.krate();
        let mut annotator = Annotator {
            tcx: tcx,
            index: self,
            parent_stab: None,
            parent_depr: None,
            in_trait_impl: false,
        };
        annotator.annotate(ast::CRATE_NODE_ID, &krate.attrs, krate.span, AnnotationKind::Required,
                           |v| intravisit::walk_crate(v, krate));
    }

    pub fn new(hir_map: &hir_map::Map) -> Index<'tcx> {
        let _task = hir_map.dep_graph.in_task(DepNode::StabilityIndex);
        let krate = hir_map.krate();

        let mut is_staged_api = false;
        for attr in &krate.attrs {
            if attr.name() == "stable" || attr.name() == "unstable" {
                is_staged_api = true;
                break
            }
        }

        let mut staged_api = FxHashMap();
        staged_api.insert(LOCAL_CRATE, is_staged_api);
        Index {
            staged_api: staged_api,
            stab_map: DefIdMap(),
            depr_map: DefIdMap(),
            active_features: FxHashSet(),
            used_features: FxHashMap(),
        }
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
pub fn check_unstable_api_usage<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut checker = Checker { tcx: tcx };
    tcx.visit_all_item_likes_in_krate(DepNode::StabilityCheck, &mut checker.as_deep_visitor());
}

struct Checker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    // (See issue #38412)
    fn skip_stability_check_due_to_privacy(self, def_id: DefId) -> bool {
        let visibility = {
            // Check if `def_id` is a trait method.
            match self.sess.cstore.associated_item(def_id) {
                Some(ty::AssociatedItem { container: ty::TraitContainer(trait_def_id), .. }) => {
                    // Trait methods do not declare visibility (even
                    // for visibility info in cstore). Use containing
                    // trait instead, so methods of pub traits are
                    // themselves considered pub.
                    self.sess.cstore.visibility(trait_def_id)
                }
                _ => {
                    // Otherwise, cstore info works directly.
                    self.sess.cstore.visibility(def_id)
                }
            }
        };

        match visibility {
            // must check stability for pub items.
            ty::Visibility::Public => false,

            // these are not visible outside crate; therefore
            // stability markers are irrelevant, if even present.
            ty::Visibility::Restricted(..) |
            ty::Visibility::Invisible => true,
        }
    }

    pub fn check_stability(self, def_id: DefId, id: NodeId, span: Span) {
        if self.sess.codemap().span_allows_unstable(span) {
            debug!("stability: \
                    skipping span={:?} since it is internal", span);
            return;
        }

        let lint_deprecated = |note: Option<Symbol>| {
            let msg = if let Some(note) = note {
                format!("use of deprecated item: {}", note)
            } else {
                format!("use of deprecated item")
            };

            self.sess.add_lint(lint::builtin::DEPRECATED, id, span, msg);
        };

        // Deprecated attributes apply in-crate and cross-crate.
        if let Some(depr_entry) = self.lookup_deprecation_entry(def_id) {
            let skip = if id == ast::DUMMY_NODE_ID {
                true
            } else {
                let parent_def_id = self.map.local_def_id(self.map.get_parent(id));
                self.lookup_deprecation_entry(parent_def_id).map_or(false, |parent_depr| {
                    parent_depr.same_origin(&depr_entry)
                })
            };

            if !skip {
                lint_deprecated(depr_entry.attr.note);
            }
        }

        let is_staged_api = *self.stability.borrow_mut().staged_api.entry(def_id.krate)
            .or_insert_with(|| self.sess.cstore.is_staged_api(def_id.krate));
        if !is_staged_api {
            return;
        }

        let stability = self.lookup_stability(def_id);
        debug!("stability: \
                inspecting def_id={:?} span={:?} of stability={:?}", def_id, span, stability);

        if let Some(&Stability{rustc_depr: Some(attr::RustcDeprecation { reason, .. }), ..})
                = stability {
            if id != ast::DUMMY_NODE_ID {
                lint_deprecated(Some(reason));
            }
        }

        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !def_id.is_local();
        if !cross_crate {
            return
        }

        if let Some(&Stability { ref level, ref feature, .. }) = stability {
            self.stability.borrow_mut().used_features.insert(feature.clone(), level.clone());
        }

        // Issue 38412: private items lack stability markers.
        if self.skip_stability_check_due_to_privacy(def_id) {
            return
        }

        match stability {
            Some(&Stability { level: attr::Unstable {ref reason, issue}, ref feature, .. }) => {
                if !self.stability.borrow().active_features.contains(feature) {
                    let msg = match *reason {
                        Some(ref r) => format!("use of unstable library feature '{}': {}",
                                               &feature.as_str(), &r),
                        None => format!("use of unstable library feature '{}'", &feature)
                    };
                    emit_feature_err(&self.sess.parse_sess, &feature.as_str(), span,
                                     GateIssue::Library(Some(issue)), &msg);
                }
            }
            Some(_) => {
                // Stable APIs are always ok to call and deprecated APIs are
                // handled by the lint emitting logic above.
            }
            None => {
                span_bug!(span, "encountered unmarked API");
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Checker<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.map)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            hir::ItemExternCrate(_) => {
                // compiler-generated `extern crate` items have a dummy span.
                if item.span == DUMMY_SP { return }

                let cnum = match self.tcx.sess.cstore.extern_mod_stmt_cnum(item.id) {
                    Some(cnum) => cnum,
                    None => return,
                };
                let def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
                self.tcx.check_stability(def_id, item.id, item.span);
            }

            // For implementations of traits, check the stability of each item
            // individually as it's possible to have a stable trait with unstable
            // items.
            hir::ItemImpl(.., Some(ref t), _, ref impl_item_refs) => {
                if let Def::Trait(trait_did) = t.path.def {
                    for impl_item_ref in impl_item_refs {
                        let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
                        let trait_item_def_id = self.tcx.associated_items(trait_did)
                            .find(|item| item.name == impl_item.name).map(|item| item.def_id);
                        if let Some(def_id) = trait_item_def_id {
                            // Pass `DUMMY_NODE_ID` to skip deprecation warnings.
                            self.tcx.check_stability(def_id, ast::DUMMY_NODE_ID, impl_item.span);
                        }
                    }
                }
            }

            _ => (/* pass */)
        }
        intravisit::walk_item(self, item);
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, id: ast::NodeId) {
        match path.def {
            Def::PrimTy(..) | Def::SelfTy(..) | Def::Err => {}
            _ => self.tcx.check_stability(path.def.def_id(), id, path.span)
        }
        intravisit::walk_path(self, path)
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Lookup the stability for a node, loading external crate
    /// metadata as necessary.
    pub fn lookup_stability(self, id: DefId) -> Option<&'gcx Stability> {
        if let Some(st) = self.stability.borrow().stab_map.get(&id) {
            return *st;
        }

        let st = self.lookup_stability_uncached(id);
        self.stability.borrow_mut().stab_map.insert(id, st);
        st
    }

    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        self.lookup_deprecation_entry(id).map(|depr| depr.attr)
    }

    pub fn lookup_deprecation_entry(self, id: DefId) -> Option<DeprecationEntry> {
        if let Some(depr) = self.stability.borrow().depr_map.get(&id) {
            return depr.clone();
        }

        let depr = self.lookup_deprecation_uncached(id);
        self.stability.borrow_mut().depr_map.insert(id, depr.clone());
        depr
    }

    fn lookup_stability_uncached(self, id: DefId) -> Option<&'gcx Stability> {
        debug!("lookup(id={:?})", id);
        if id.is_local() {
            None // The stability cache is filled partially lazily
        } else {
            self.sess.cstore.stability(id).map(|st| self.intern_stability(st))
        }
    }

    fn lookup_deprecation_uncached(self, id: DefId) -> Option<DeprecationEntry> {
        debug!("lookup(id={:?})", id);
        if id.is_local() {
            None // The stability cache is filled partially lazily
        } else {
            self.sess.cstore.deprecation(id).map(DeprecationEntry::external)
        }
    }
}

/// Given the list of enabled features that were not language features (i.e. that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                 access_levels: &AccessLevels) {
    let sess = &tcx.sess;

    if tcx.stability.borrow().staged_api[&LOCAL_CRATE] && tcx.sess.features.borrow().staged_api {
        let _task = tcx.dep_graph.in_task(DepNode::StabilityIndex);
        let krate = tcx.map.krate();
        let mut missing = MissingStabilityAnnotations {
            tcx: tcx,
            access_levels: access_levels,
        };
        missing.check_missing_stability(ast::CRATE_NODE_ID, krate.span);
        intravisit::walk_crate(&mut missing, krate);
        krate.visit_all_item_likes(&mut missing.as_deep_visitor());
    }

    let ref declared_lib_features = sess.features.borrow().declared_lib_features;
    let mut remaining_lib_features: FxHashMap<Symbol, Span>
        = declared_lib_features.clone().into_iter().collect();

    fn format_stable_since_msg(version: &str) -> String {
        format!("this feature has been stable since {}. Attribute no longer needed", version)
    }

    for &(ref stable_lang_feature, span) in &sess.features.borrow().declared_stable_lang_features {
        let version = find_lang_feature_accepted_version(&stable_lang_feature.as_str())
            .expect("unexpectedly couldn't find version feature was stabilized");
        sess.add_lint(lint::builtin::STABLE_FEATURES,
                      ast::CRATE_NODE_ID,
                      span,
                      format_stable_since_msg(version));
    }

    let index = tcx.stability.borrow();
    for (used_lib_feature, level) in &index.used_features {
        match remaining_lib_features.remove(used_lib_feature) {
            Some(span) => {
                if let &attr::StabilityLevel::Stable { since: ref version } = level {
                    sess.add_lint(lint::builtin::STABLE_FEATURES,
                                  ast::CRATE_NODE_ID,
                                  span,
                                  format_stable_since_msg(&version.as_str()));
                }
            }
            None => ( /* used but undeclared, handled during the previous ast visit */ )
        }
    }

    for &span in remaining_lib_features.values() {
        sess.add_lint(lint::builtin::UNUSED_FEATURES,
                      ast::CRATE_NODE_ID,
                      span,
                      "unused or unknown feature".to_string());
    }
}
