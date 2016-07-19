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
use session::Session;
use lint;
use middle::cstore::LOCAL_CRATE;
use hir::def::Def;
use hir::def_id::{CRATE_DEF_INDEX, DefId};
use ty::{self, TyCtxt};
use middle::privacy::AccessLevels;
use syntax::parse::token::InternedString;
use syntax_pos::{Span, DUMMY_SP};
use syntax::ast;
use syntax::ast::{NodeId, Attribute};
use syntax::feature_gate::{GateIssue, emit_feature_err, find_lang_feature_accepted_version};
use syntax::attr::{self, Stability, Deprecation, AttrMetaMethods};
use util::nodemap::{DefIdMap, FnvHashSet, FnvHashMap};

use hir;
use hir::{Item, Generics, StructField, Variant, PatKind};
use hir::intravisit::{self, Visitor};
use hir::pat_util::EnumerateAndAdjustIterator;

use std::mem::replace;
use std::cmp::Ordering;
use std::ops::Deref;

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

/// A stability index, giving the stability level for items and methods.
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    stab_map: DefIdMap<Option<&'tcx Stability>>,
    depr_map: DefIdMap<Option<Deprecation>>,

    /// Maps for each crate whether it is part of the staged API.
    staged_api: FnvHashMap<ast::CrateNum, bool>
}

// A private tree-walker for producing an Index.
struct Annotator<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    index: &'a mut Index<'tcx>,
    parent_stab: Option<&'tcx Stability>,
    parent_depr: Option<Deprecation>,
    access_levels: &'a AccessLevels,
    in_trait_impl: bool,
}

impl<'a, 'tcx: 'a> Annotator<'a, 'tcx> {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate<F>(&mut self, id: NodeId, attrs: &[Attribute],
                   item_sp: Span, kind: AnnotationKind, visit_children: F)
        where F: FnOnce(&mut Annotator)
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
                if let (&Some(attr::RustcDeprecation {since: ref dep_since, ..}),
                        &attr::Stable {since: ref stab_since}) = (&stab.rustc_depr, &stab.level) {
                    // Explicit version of iter::order::lt to handle parse errors properly
                    for (dep_v, stab_v) in dep_since.split(".").zip(stab_since.split(".")) {
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
                let mut is_error = kind == AnnotationKind::Required &&
                                   self.access_levels.is_reachable(id) &&
                                   !self.tcx.sess.opts.test;
                if let Some(stab) = self.parent_stab {
                    if stab.level.is_unstable() {
                        let def_id = self.tcx.map.local_def_id(id);
                        self.index.stab_map.insert(def_id, Some(stab));
                        is_error = false;
                    }
                }
                if is_error {
                    self.tcx.sess.span_err(item_sp, "This node does not have \
                                                     a stability attribute");
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
                self.index.depr_map.insert(def_id, Some(depr.clone()));

                let orig_parent_depr = replace(&mut self.parent_depr, Some(depr));
                visit_children(self);
                self.parent_depr = orig_parent_depr;
            } else if let Some(depr) = self.parent_depr.clone() {
                let def_id = self.tcx.map.local_def_id(id);
                self.index.depr_map.insert(def_id, Some(depr));
                visit_children(self);
            } else {
                visit_children(self);
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for Annotator<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        let tcx = self.tcx;
        self.visit_item(tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, i: &Item) {
        let orig_in_trait_impl = self.in_trait_impl;
        let mut kind = AnnotationKind::Required;
        match i.node {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemImpl(_, _, _, None, _, _) | hir::ItemForeignMod(..) => {
                self.in_trait_impl = false;
                kind = AnnotationKind::Container;
            }
            hir::ItemImpl(_, _, _, Some(_), _, _) => {
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

    fn visit_trait_item(&mut self, ti: &hir::TraitItem) {
        self.annotate(ti.id, &ti.attrs, ti.span, AnnotationKind::Required, |v| {
            intravisit::walk_trait_item(v, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &hir::ImplItem) {
        let kind = if self.in_trait_impl {
            AnnotationKind::Prohibited
        } else {
            AnnotationKind::Required
        };
        self.annotate(ii.id, &ii.attrs, ii.span, kind, |v| {
            intravisit::walk_impl_item(v, ii);
        });
    }

    fn visit_variant(&mut self, var: &Variant, g: &'v Generics, item_id: NodeId) {
        self.annotate(var.node.data.id(), &var.node.attrs, var.span, AnnotationKind::Required, |v| {
            intravisit::walk_variant(v, var, g, item_id);
        })
    }

    fn visit_struct_field(&mut self, s: &StructField) {
        self.annotate(s.id, &s.attrs, s.span, AnnotationKind::Required, |v| {
            intravisit::walk_struct_field(v, s);
        });
    }

    fn visit_foreign_item(&mut self, i: &hir::ForeignItem) {
        self.annotate(i.id, &i.attrs, i.span, AnnotationKind::Required, |v| {
            intravisit::walk_foreign_item(v, i);
        });
    }

    fn visit_macro_def(&mut self, md: &'v hir::MacroDef) {
        if md.imported_from.is_none() {
            self.annotate(md.id, &md.attrs, md.span, AnnotationKind::Required, |_| {});
        }
    }
}

impl<'a, 'tcx> Index<'tcx> {
    /// Construct the stability index for a crate being compiled.
    pub fn build(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, access_levels: &AccessLevels) {
        let _task = tcx.dep_graph.in_task(DepNode::StabilityIndex);
        let krate = tcx.map.krate();
        let mut annotator = Annotator {
            tcx: tcx,
            index: self,
            parent_stab: None,
            parent_depr: None,
            access_levels: access_levels,
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

        let mut staged_api = FnvHashMap();
        staged_api.insert(LOCAL_CRATE, is_staged_api);
        Index {
            staged_api: staged_api,
            stab_map: DefIdMap(),
            depr_map: DefIdMap(),
        }
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors. Returns a list of all
/// features used.
pub fn check_unstable_api_usage<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                          -> FnvHashMap<InternedString, attr::StabilityLevel> {
    let _task = tcx.dep_graph.in_task(DepNode::StabilityCheck);
    let ref active_lib_features = tcx.sess.features.borrow().declared_lib_features;

    // Put the active features into a map for quick lookup
    let active_features = active_lib_features.iter().map(|&(ref s, _)| s.clone()).collect();

    let mut checker = Checker {
        tcx: tcx,
        active_features: active_features,
        used_features: FnvHashMap(),
        in_skip_block: 0,
    };
    intravisit::walk_crate(&mut checker, tcx.map.krate());

    checker.used_features
}

struct Checker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    active_features: FnvHashSet<InternedString>,
    used_features: FnvHashMap<InternedString, attr::StabilityLevel>,
    // Within a block where feature gate checking can be skipped.
    in_skip_block: u32,
}

impl<'a, 'tcx> Checker<'a, 'tcx> {
    fn check(&mut self, id: DefId, span: Span,
             stab: &Option<&Stability>, _depr: &Option<Deprecation>) {
        if !is_staged_api(self.tcx, id) {
            return;
        }
        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !id.is_local();
        if !cross_crate {
            return
        }

        // We don't need to check for stability - presumably compiler generated code.
        if self.in_skip_block > 0 {
            return;
        }

        match *stab {
            Some(&Stability { level: attr::Unstable {ref reason, issue}, ref feature, .. }) => {
                self.used_features.insert(feature.clone(),
                                          attr::Unstable { reason: reason.clone(), issue: issue });

                if !self.active_features.contains(feature) {
                    let msg = match *reason {
                        Some(ref r) => format!("use of unstable library feature '{}': {}",
                                               &feature, &r),
                        None => format!("use of unstable library feature '{}'", &feature)
                    };
                    emit_feature_err(&self.tcx.sess.parse_sess.span_diagnostic,
                                      &feature, span, GateIssue::Library(Some(issue)), &msg);
                }
            }
            Some(&Stability { ref level, ref feature, .. }) => {
                self.used_features.insert(feature.clone(), level.clone());

                // Stable APIs are always ok to call and deprecated APIs are
                // handled by a lint.
            }
            None => {
                // This is an 'unmarked' API, which should not exist
                // in the standard library.
                if self.tcx.sess.features.borrow().unmarked_api {
                    self.tcx.sess.struct_span_warn(span, "use of unmarked library feature")
                                 .span_note(span, "this is either a bug in the library you are \
                                                   using or a bug in the compiler - please \
                                                   report it in both places")
                                 .emit()
                } else {
                    self.tcx.sess.struct_span_err(span, "use of unmarked library feature")
                                 .span_note(span, "this is either a bug in the library you are \
                                                   using or a bug in the compiler - please \
                                                   report it in both places")
                                 .span_note(span, "use #![feature(unmarked_api)] in the \
                                                   crate attributes to override this")
                                 .emit()
                }
            }
        }
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for Checker<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        let tcx = self.tcx;
        self.visit_item(tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &hir::Item) {
        // When compiling with --test we don't enforce stability on the
        // compiler-generated test module, demarcated with `DUMMY_SP` plus the
        // name `__test`
        if item.span == DUMMY_SP && item.name.as_str() == "__test" { return }

        check_item(self.tcx, item, true,
                   &mut |id, sp, stab, depr| self.check(id, sp, stab, depr));
        intravisit::walk_item(self, item);
    }

    fn visit_expr(&mut self, ex: &hir::Expr) {
        check_expr(self.tcx, ex,
                   &mut |id, sp, stab, depr| self.check(id, sp, stab, depr));
        intravisit::walk_expr(self, ex);
    }

    fn visit_path(&mut self, path: &hir::Path, id: ast::NodeId) {
        check_path(self.tcx, path, id,
                   &mut |id, sp, stab, depr| self.check(id, sp, stab, depr));
        intravisit::walk_path(self, path)
    }

    fn visit_path_list_item(&mut self, prefix: &hir::Path, item: &hir::PathListItem) {
        check_path_list_item(self.tcx, item,
                   &mut |id, sp, stab, depr| self.check(id, sp, stab, depr));
        intravisit::walk_path_list_item(self, prefix, item)
    }

    fn visit_pat(&mut self, pat: &hir::Pat) {
        check_pat(self.tcx, pat,
                  &mut |id, sp, stab, depr| self.check(id, sp, stab, depr));
        intravisit::walk_pat(self, pat)
    }

    fn visit_block(&mut self, b: &hir::Block) {
        let old_skip_count = self.in_skip_block;
        match b.rules {
            hir::BlockCheckMode::PushUnstableBlock => {
                self.in_skip_block += 1;
            }
            hir::BlockCheckMode::PopUnstableBlock => {
                self.in_skip_block = self.in_skip_block.checked_sub(1).unwrap();
            }
            _ => {}
        }
        intravisit::walk_block(self, b);
        self.in_skip_block = old_skip_count;
    }
}

/// Helper for discovering nodes to check for stability
pub fn check_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            item: &hir::Item,
                            warn_about_defns: bool,
                            cb: &mut FnMut(DefId, Span,
                                           &Option<&Stability>,
                                           &Option<Deprecation>)) {
    match item.node {
        hir::ItemExternCrate(_) => {
            // compiler-generated `extern crate` items have a dummy span.
            if item.span == DUMMY_SP { return }

            let cnum = match tcx.sess.cstore.extern_mod_stmt_cnum(item.id) {
                Some(cnum) => cnum,
                None => return,
            };
            let id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
            maybe_do_stability_check(tcx, id, item.span, cb);
        }

        // For implementations of traits, check the stability of each item
        // individually as it's possible to have a stable trait with unstable
        // items.
        hir::ItemImpl(_, _, _, Some(ref t), _, ref impl_items) => {
            let trait_did = tcx.expect_def(t.ref_id).def_id();
            let trait_items = tcx.trait_items(trait_did);

            for impl_item in impl_items {
                let item = trait_items.iter().find(|item| {
                    item.name() == impl_item.name
                }).unwrap();
                if warn_about_defns {
                    maybe_do_stability_check(tcx, item.def_id(), impl_item.span, cb);
                }
            }
        }

        _ => (/* pass */)
    }
}

/// Helper for discovering nodes to check for stability
pub fn check_expr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, e: &hir::Expr,
                            cb: &mut FnMut(DefId, Span,
                                           &Option<&Stability>,
                                           &Option<Deprecation>)) {
    let span;
    let id = match e.node {
        hir::ExprMethodCall(i, _, _) => {
            span = i.span;
            let method_call = ty::MethodCall::expr(e.id);
            tcx.tables.borrow().method_map[&method_call].def_id
        }
        hir::ExprField(ref base_e, ref field) => {
            span = field.span;
            match tcx.expr_ty_adjusted(base_e).sty {
                ty::TyStruct(def, _) => def.struct_variant().field_named(field.node).did,
                _ => span_bug!(e.span,
                               "stability::check_expr: named field access on non-struct")
            }
        }
        hir::ExprTupField(ref base_e, ref field) => {
            span = field.span;
            match tcx.expr_ty_adjusted(base_e).sty {
                ty::TyStruct(def, _) => def.struct_variant().fields[field.node].did,
                ty::TyTuple(..) => return,
                _ => span_bug!(e.span,
                               "stability::check_expr: unnamed field access on \
                                something other than a tuple or struct")
            }
        }
        hir::ExprStruct(_, ref expr_fields, _) => {
            let type_ = tcx.expr_ty(e);
            match type_.sty {
                ty::TyStruct(def, _) => {
                    // check the stability of each field that appears
                    // in the construction expression.
                    for field in expr_fields {
                        let did = def.struct_variant()
                            .field_named(field.name.node)
                            .did;
                        maybe_do_stability_check(tcx, did, field.span, cb);
                    }

                    // we're done.
                    return
                }
                // we don't look at stability attributes on
                // struct-like enums (yet...), but it's definitely not
                // a bug to have construct one.
                ty::TyEnum(..) => return,
                _ => {
                    span_bug!(e.span,
                              "stability::check_expr: struct construction \
                               of non-struct, type {:?}",
                              type_);
                }
            }
        }
        _ => return
    };

    maybe_do_stability_check(tcx, id, span, cb);
}

pub fn check_path<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            path: &hir::Path, id: ast::NodeId,
                            cb: &mut FnMut(DefId, Span,
                                           &Option<&Stability>,
                                           &Option<Deprecation>)) {
    // Paths in import prefixes may have no resolution.
    match tcx.expect_def_or_none(id) {
        Some(Def::PrimTy(..)) => {}
        Some(Def::SelfTy(..)) => {}
        Some(def) => {
            maybe_do_stability_check(tcx, def.def_id(), path.span, cb);
        }
        None => {}
    }
}

pub fn check_path_list_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      item: &hir::PathListItem,
                                      cb: &mut FnMut(DefId, Span,
                                                     &Option<&Stability>,
                                                     &Option<Deprecation>)) {
    match tcx.expect_def(item.node.id()) {
        Def::PrimTy(..) => {}
        def => {
            maybe_do_stability_check(tcx, def.def_id(), item.span, cb);
        }
    }
}

pub fn check_pat<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, pat: &hir::Pat,
                           cb: &mut FnMut(DefId, Span,
                                          &Option<&Stability>,
                                          &Option<Deprecation>)) {
    debug!("check_pat(pat = {:?})", pat);
    if is_internal(tcx, pat.span) { return; }

    let v = match tcx.pat_ty_opt(pat) {
        Some(&ty::TyS { sty: ty::TyStruct(def, _), .. }) => def.struct_variant(),
        Some(_) | None => return,
    };
    match pat.node {
        // Foo(a, b, c)
        PatKind::TupleStruct(_, ref pat_fields, ddpos) => {
            for (i, field) in pat_fields.iter().enumerate_and_adjust(v.fields.len(), ddpos) {
                maybe_do_stability_check(tcx, v.fields[i].did, field.span, cb)
            }
        }
        // Foo { a, b, c }
        PatKind::Struct(_, ref pat_fields, _) => {
            for field in pat_fields {
                let did = v.field_named(field.node.name).did;
                maybe_do_stability_check(tcx, did, field.span, cb);
            }
        }
        // everything else is fine.
        _ => {}
    }
}

fn maybe_do_stability_check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      id: DefId, span: Span,
                                      cb: &mut FnMut(DefId, Span,
                                                     &Option<&Stability>,
                                                     &Option<Deprecation>)) {
    if is_internal(tcx, span) {
        debug!("maybe_do_stability_check: \
                skipping span={:?} since it is internal", span);
        return;
    }
    let (stability, deprecation) = if is_staged_api(tcx, id) {
        (tcx.lookup_stability(id), None)
    } else {
        (None, tcx.lookup_deprecation(id))
    };
    debug!("maybe_do_stability_check: \
            inspecting id={:?} span={:?} of stability={:?}", id, span, stability);
    cb(id, span, &stability, &deprecation);
}

fn is_internal<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, span: Span) -> bool {
    tcx.sess.codemap().span_allows_unstable(span)
}

fn is_staged_api<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, id: DefId) -> bool {
    match tcx.trait_item_of_item(id) {
        Some(ty::MethodTraitItemId(trait_method_id))
            if trait_method_id != id => {
                is_staged_api(tcx, trait_method_id)
            }
        _ => {
            *tcx.stability.borrow_mut().staged_api.entry(id.krate).or_insert_with(
                || tcx.sess.cstore.is_staged_api(id.krate))
        }
    }
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    /// Lookup the stability for a node, loading external crate
    /// metadata as necessary.
    pub fn lookup_stability(self, id: DefId) -> Option<&'tcx Stability> {
        if let Some(st) = self.stability.borrow().stab_map.get(&id) {
            return *st;
        }

        let st = self.lookup_stability_uncached(id);
        self.stability.borrow_mut().stab_map.insert(id, st);
        st
    }

    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        if let Some(depr) = self.stability.borrow().depr_map.get(&id) {
            return depr.clone();
        }

        let depr = self.lookup_deprecation_uncached(id);
        self.stability.borrow_mut().depr_map.insert(id, depr.clone());
        depr
    }

    fn lookup_stability_uncached(self, id: DefId) -> Option<&'tcx Stability> {
        debug!("lookup(id={:?})", id);
        if id.is_local() {
            None // The stability cache is filled partially lazily
        } else {
            self.sess.cstore.stability(id).map(|st| self.intern_stability(st))
        }
    }

    fn lookup_deprecation_uncached(self, id: DefId) -> Option<Deprecation> {
        debug!("lookup(id={:?})", id);
        if id.is_local() {
            None // The stability cache is filled partially lazily
        } else {
            self.sess.cstore.deprecation(id)
        }
    }
}

/// Given the list of enabled features that were not language features (i.e. that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features(sess: &Session,
                                       lib_features_used: &FnvHashMap<InternedString,
                                                                      attr::StabilityLevel>) {
    let ref declared_lib_features = sess.features.borrow().declared_lib_features;
    let mut remaining_lib_features: FnvHashMap<InternedString, Span>
        = declared_lib_features.clone().into_iter().collect();

    fn format_stable_since_msg(version: &str) -> String {
        format!("this feature has been stable since {}. Attribute no longer needed", version)
    }

    for &(ref stable_lang_feature, span) in &sess.features.borrow().declared_stable_lang_features {
        let version = find_lang_feature_accepted_version(stable_lang_feature.deref())
            .expect("unexpectedly couldn't find version feature was stabilized");
        sess.add_lint(lint::builtin::STABLE_FEATURES,
                      ast::CRATE_NODE_ID,
                      span,
                      format_stable_since_msg(version));
    }

    for (used_lib_feature, level) in lib_features_used {
        match remaining_lib_features.remove(used_lib_feature) {
            Some(span) => {
                if let &attr::StabilityLevel::Stable { since: ref version } = level {
                    sess.add_lint(lint::builtin::STABLE_FEATURES,
                                  ast::CRATE_NODE_ID,
                                  span,
                                  format_stable_since_msg(version.deref()));
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
