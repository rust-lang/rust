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

use lint;
use hir::def::Def;
use hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId, LOCAL_CRATE};
use ty::{self, TyCtxt};
use middle::privacy::AccessLevels;
use syntax::symbol::Symbol;
use syntax_pos::{Span, DUMMY_SP};
use syntax::ast;
use syntax::ast::{NodeId, Attribute};
use syntax::feature_gate::{GateIssue, emit_feature_err, find_lang_feature_accepted_version};
use syntax::attr::{self, Stability, Deprecation};
use util::nodemap::{FxHashSet, FxHashMap};

use hir;
use hir::{Item, Generics, StructField, Variant, HirId};
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
    origin: Option<HirId>,
}

impl_stable_hash_for!(struct self::DeprecationEntry {
    attr,
    origin
});

impl DeprecationEntry {
    fn local(attr: Deprecation, id: HirId) -> DeprecationEntry {
        DeprecationEntry {
            attr,
            origin: Some(id),
        }
    }

    pub fn external(attr: Deprecation) -> DeprecationEntry {
        DeprecationEntry {
            attr,
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
    stab_map: FxHashMap<HirId, &'tcx Stability>,
    depr_map: FxHashMap<HirId, DeprecationEntry>,

    /// Maps for each crate whether it is part of the staged API.
    staged_api: FxHashMap<CrateNum, bool>,

    /// Features enabled for this crate.
    active_features: FxHashSet<Symbol>,
}

impl_stable_hash_for!(struct self::Index<'tcx> {
    stab_map,
    depr_map,
    staged_api,
    active_features
});

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
        if self.tcx.sess.features.borrow().staged_api {
            // This crate explicitly wants staged API.
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

                let hir_id = self.tcx.hir.node_to_hir_id(id);
                self.index.stab_map.insert(hir_id, stab);

                let orig_parent_stab = replace(&mut self.parent_stab, Some(stab));
                visit_children(self);
                self.parent_stab = orig_parent_stab;
            } else {
                debug!("annotate: not found, parent = {:?}", self.parent_stab);
                if let Some(stab) = self.parent_stab {
                    if stab.level.is_unstable() {
                        let hir_id = self.tcx.hir.node_to_hir_id(id);
                        self.index.stab_map.insert(hir_id, stab);
                    }
                }
                visit_children(self);
            }
        } else {
            // Emit errors for non-staged-api crates.
            for attr in attrs {
                let tag = unwrap_or!(attr.name(), continue);
                if tag == "unstable" || tag == "stable" || tag == "rustc_deprecated" {
                    attr::mark_used(attr);
                    self.tcx.sess.span_err(attr.span(), "stability attributes may not be used \
                                                         outside of the standard library");
                }
            }

            // Propagate unstability.  This can happen even for non-staged-api crates in case
            // -Zforce-unstable-if-unmarked is set.
            if let Some(stab) = self.parent_stab {
                if stab.level.is_unstable() {
                    let hir_id = self.tcx.hir.node_to_hir_id(id);
                    self.index.stab_map.insert(hir_id, stab);
                }
            }

            if let Some(depr) = attr::find_deprecation(self.tcx.sess.diagnostic(), attrs, item_sp) {
                if kind == AnnotationKind::Prohibited {
                    self.tcx.sess.span_err(item_sp, "This deprecation annotation is useless");
                }

                // `Deprecation` is just two pointers, no need to intern it
                let hir_id = self.tcx.hir.node_to_hir_id(id);
                let depr_entry = DeprecationEntry::local(depr, hir_id);
                self.index.depr_map.insert(hir_id, depr_entry.clone());

                let orig_parent_depr = replace(&mut self.parent_depr,
                                               Some(depr_entry));
                visit_children(self);
                self.parent_depr = orig_parent_depr;
            } else if let Some(parent_depr) = self.parent_depr.clone() {
                let hir_id = self.tcx.hir.node_to_hir_id(id);
                self.index.depr_map.insert(hir_id, parent_depr);
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
        NestedVisitorMap::All(&self.tcx.hir)
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
        let hir_id = self.tcx.hir.node_to_hir_id(id);
        let stab = self.tcx.stability().local_stability(hir_id);
        let is_error = !self.tcx.sess.opts.test &&
                        stab.is_none() &&
                        self.access_levels.is_reachable(id);
        if is_error {
            self.tcx.sess.span_err(span, "This node does not have a stability attribute");
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir)
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
        let impl_def_id = self.tcx.hir.local_def_id(self.tcx.hir.get_parent(ii.id));
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
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Index<'tcx> {
        let is_staged_api =
            tcx.sess.opts.debugging_opts.force_unstable_if_unmarked ||
            tcx.sess.features.borrow().staged_api;
        let mut staged_api = FxHashMap();
        staged_api.insert(LOCAL_CRATE, is_staged_api);
        let mut index = Index {
            staged_api,
            stab_map: FxHashMap(),
            depr_map: FxHashMap(),
            active_features: FxHashSet(),
        };

        let ref active_lib_features = tcx.sess.features.borrow().declared_lib_features;

        // Put the active features into a map for quick lookup
        index.active_features = active_lib_features.iter().map(|&(ref s, _)| s.clone()).collect();

        {
            let krate = tcx.hir.krate();
            let mut annotator = Annotator {
                tcx,
                index: &mut index,
                parent_stab: None,
                parent_depr: None,
                in_trait_impl: false,
            };

            // If the `-Z force-unstable-if-unmarked` flag is passed then we provide
            // a parent stability annotation which indicates that this is private
            // with the `rustc_private` feature. This is intended for use when
            // compiling librustc crates themselves so we can leverage crates.io
            // while maintaining the invariant that all sysroot crates are unstable
            // by default and are unable to be used.
            if tcx.sess.opts.debugging_opts.force_unstable_if_unmarked {
                let reason = "this crate is being loaded from the sysroot, and \
                              unstable location; did you mean to load this crate \
                              from crates.io via `Cargo.toml` instead?";
                let stability = tcx.intern_stability(Stability {
                    level: attr::StabilityLevel::Unstable {
                        reason: Some(Symbol::intern(reason)),
                        issue: 27812,
                    },
                    feature: Symbol::intern("rustc_private"),
                    rustc_depr: None,
                    rustc_const_unstable: None,
                });
                annotator.parent_stab = Some(stability);
            }

            annotator.annotate(ast::CRATE_NODE_ID,
                               &krate.attrs,
                               krate.span,
                               AnnotationKind::Required,
                               |v| intravisit::walk_crate(v, krate));
        }
        return index
    }

    pub fn local_stability(&self, id: HirId) -> Option<&'tcx Stability> {
        self.stab_map.get(&id).cloned()
    }

    pub fn local_deprecation_entry(&self, id: HirId) -> Option<DeprecationEntry> {
        self.depr_map.get(&id).cloned()
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
pub fn check_unstable_api_usage<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut checker = Checker { tcx: tcx };
    tcx.hir.krate().visit_all_item_likes(&mut checker.as_deep_visitor());
}

struct Checker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    // (See issue #38412)
    fn skip_stability_check_due_to_privacy(self, mut def_id: DefId) -> bool {
        // Check if `def_id` is a trait method.
        match self.describe_def(def_id) {
            Some(Def::Method(_)) |
            Some(Def::AssociatedTy(_)) |
            Some(Def::AssociatedConst(_)) => {
                match self.associated_item(def_id).container {
                    ty::TraitContainer(trait_def_id) => {
                        // Trait methods do not declare visibility (even
                        // for visibility info in cstore). Use containing
                        // trait instead, so methods of pub traits are
                        // themselves considered pub.
                        def_id = trait_def_id;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        let visibility = self.visibility(def_id);

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
        if span.allows_unstable() {
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

            self.lint_node(lint::builtin::DEPRECATED, id, span, &msg);
        };

        // Deprecated attributes apply in-crate and cross-crate.
        if let Some(depr_entry) = self.lookup_deprecation_entry(def_id) {
            let skip = if id == ast::DUMMY_NODE_ID {
                true
            } else {
                let parent_def_id = self.hir.local_def_id(self.hir.get_parent(id));
                self.lookup_deprecation_entry(parent_def_id).map_or(false, |parent_depr| {
                    parent_depr.same_origin(&depr_entry)
                })
            };

            if !skip {
                lint_deprecated(depr_entry.attr.note);
            }
        }

        let is_staged_api = self.lookup_stability(DefId {
            index: CRATE_DEF_INDEX,
            ..def_id
        }).is_some();
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

        // Issue 38412: private items lack stability markers.
        if self.skip_stability_check_due_to_privacy(def_id) {
            return
        }

        match stability {
            Some(&Stability { level: attr::Unstable {ref reason, issue}, ref feature, .. }) => {
                if self.stability().active_features.contains(feature) {
                    return
                }

                // When we're compiling the compiler itself we may pull in
                // crates from crates.io, but those crates may depend on other
                // crates also pulled in from crates.io. We want to ideally be
                // able to compile everything without requiring upstream
                // modifications, so in the case that this looks like a
                // rustc_private crate (e.g. a compiler crate) and we also have
                // the `-Z force-unstable-if-unmarked` flag present (we're
                // compiling a compiler crate), then let this missing feature
                // annotation slide.
                if *feature == "rustc_private" && issue == 27812 {
                    if self.sess.opts.debugging_opts.force_unstable_if_unmarked {
                        return
                    }
                }

                let msg = match *reason {
                    Some(ref r) => format!("use of unstable library feature '{}': {}",
                                           feature.as_str(), &r),
                    None => format!("use of unstable library feature '{}'", &feature)
                };
                emit_feature_err(&self.sess.parse_sess, &feature.as_str(), span,
                                 GateIssue::Library(Some(issue)), &msg);
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
        NestedVisitorMap::OnlyBodies(&self.tcx.hir)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            hir::ItemExternCrate(_) => {
                // compiler-generated `extern crate` items have a dummy span.
                if item.span == DUMMY_SP { return }

                let def_id = self.tcx.hir.local_def_id(item.id);
                let cnum = match self.tcx.extern_mod_stmt_cnum(def_id) {
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
                        let impl_item = self.tcx.hir.impl_item(impl_item_ref.id);
                        let trait_item_def_id = self.tcx.associated_items(trait_did)
                            .find(|item| item.name == impl_item.name).map(|item| item.def_id);
                        if let Some(def_id) = trait_item_def_id {
                            // Pass `DUMMY_NODE_ID` to skip deprecation warnings.
                            self.tcx.check_stability(def_id, ast::DUMMY_NODE_ID, impl_item.span);
                        }
                    }
                }
            }

            // There's no good place to insert stability check for non-Copy unions,
            // so semi-randomly perform it here in stability.rs
            hir::ItemUnion(..) if !self.tcx.sess.features.borrow().untagged_unions => {
                let def_id = self.tcx.hir.local_def_id(item.id);
                let adt_def = self.tcx.adt_def(def_id);
                let ty = self.tcx.type_of(def_id);

                if adt_def.has_dtor(self.tcx) {
                    emit_feature_err(&self.tcx.sess.parse_sess,
                                     "untagged_unions", item.span, GateIssue::Language,
                                     "unions with `Drop` implementations are unstable");
                } else {
                    let param_env = self.tcx.param_env(def_id);
                    if !param_env.can_type_implement_copy(self.tcx, ty, item.span).is_ok() {
                        emit_feature_err(&self.tcx.sess.parse_sess,
                                        "untagged_unions", item.span, GateIssue::Language,
                                        "unions with non-`Copy` fields are unstable");
                    }
                }
            }

            _ => (/* pass */)
        }
        intravisit::walk_item(self, item);
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, id: ast::NodeId) {
        match path.def {
            Def::Local(..) | Def::Upvar(..) |
            Def::PrimTy(..) | Def::SelfTy(..) | Def::Err => {}
            _ => self.tcx.check_stability(path.def.def_id(), id, path.span)
        }
        intravisit::walk_path(self, path)
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        self.lookup_deprecation_entry(id).map(|depr| depr.attr)
    }
}

/// Given the list of enabled features that were not language features (i.e. that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let sess = &tcx.sess;

    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    if tcx.stability().staged_api[&LOCAL_CRATE] {
        let krate = tcx.hir.krate();
        let mut missing = MissingStabilityAnnotations {
            tcx,
            access_levels,
        };
        missing.check_missing_stability(ast::CRATE_NODE_ID, krate.span);
        intravisit::walk_crate(&mut missing, krate);
        krate.visit_all_item_likes(&mut missing.as_deep_visitor());
    }

    let ref declared_lib_features = sess.features.borrow().declared_lib_features;
    let mut remaining_lib_features: FxHashMap<Symbol, Span>
        = declared_lib_features.clone().into_iter().collect();
    remaining_lib_features.remove(&Symbol::intern("proc_macro"));

    for &(ref stable_lang_feature, span) in &sess.features.borrow().declared_stable_lang_features {
        let version = find_lang_feature_accepted_version(&stable_lang_feature.as_str())
            .expect("unexpectedly couldn't find version feature was stabilized");
        tcx.lint_node(lint::builtin::STABLE_FEATURES,
                      ast::CRATE_NODE_ID,
                      span,
                      &format_stable_since_msg(version));
    }

    // FIXME(#44232) the `used_features` table no longer exists, so we don't
    //               lint about unknown or unused features. We should reenable
    //               this one day!
    //
    // let index = tcx.stability();
    // for (used_lib_feature, level) in &index.used_features {
    //     remaining_lib_features.remove(used_lib_feature);
    // }
    //
    // for &span in remaining_lib_features.values() {
    //     tcx.lint_node(lint::builtin::UNUSED_FEATURES,
    //                   ast::CRATE_NODE_ID,
    //                   span,
    //                   "unused or unknown feature");
    // }
}

fn format_stable_since_msg(version: &str) -> String {
    format!("this feature has been stable since {}. Attribute no longer needed", version)
}
