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

use session::Session;
use lint;
use metadata::cstore::LOCAL_CRATE;
use middle::def;
use middle::def_id::{CRATE_DEF_INDEX, DefId};
use middle::ty;
use middle::privacy::PublicItems;
use metadata::csearch;
use syntax::parse::token::InternedString;
use syntax::codemap::{Span, DUMMY_SP};
use syntax::ast;
use syntax::ast::{NodeId, Attribute};
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax::attr::{self, Stability, AttrMetaMethods};
use util::nodemap::{DefIdMap, FnvHashSet, FnvHashMap};

use rustc_front::hir;
use rustc_front::hir::{FnDecl, Block, Crate, Item, Generics, StructField, Variant};
use rustc_front::visit::{self, FnKind, Visitor};

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

/// A stability index, giving the stability level for items and methods.
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    map: DefIdMap<Option<&'tcx Stability>>,

    /// Maps for each crate whether it is part of the staged API.
    staged_api: FnvHashMap<ast::CrateNum, bool>
}

// A private tree-walker for producing an Index.
struct Annotator<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    index: &'a mut Index<'tcx>,
    parent: Option<&'tcx Stability>,
    export_map: &'a PublicItems,
}

impl<'a, 'tcx: 'a> Annotator<'a, 'tcx> {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate<F>(&mut self, id: NodeId, use_parent: bool,
                   attrs: &Vec<Attribute>, item_sp: Span, f: F, required: bool) where
        F: FnOnce(&mut Annotator),
    {
        if self.index.staged_api[&LOCAL_CRATE] {
            debug!("annotate(id = {:?}, attrs = {:?})", id, attrs);
            match attr::find_stability(self.tcx.sess.diagnostic(), attrs, item_sp) {
                Some(mut stab) => {
                    debug!("annotate: found {:?}", stab);
                    // if parent is deprecated and we're not, inherit this by merging
                    // deprecated_since and its reason.
                    if let Some(parent_stab) = self.parent {
                        if parent_stab.depr.is_some()
                        && stab.depr.is_none() {
                            stab.depr = parent_stab.depr.clone()
                        }
                    }

                    let stab = self.tcx.intern_stability(stab);

                    // Check if deprecated_since < stable_since. If it is,
                    // this is *almost surely* an accident.
                    let deprecated_predates_stable = match (&stab.depr, &stab.level) {
                        (&Some(attr::Deprecation {since: ref dep_since, ..}),
                               &attr::Stable {since: ref stab_since}) => {
                            // explicit version of iter::order::lt to handle parse errors properly
                            let mut is_less = false;
                            for (dep_v, stab_v) in dep_since.split(".").zip(stab_since.split(".")) {
                                match (dep_v.parse::<u64>(), stab_v.parse::<u64>()) {
                                    (Ok(dep_v), Ok(stab_v)) => match dep_v.cmp(&stab_v) {
                                        Ordering::Less => {
                                            is_less = true;
                                            break;
                                        }
                                        Ordering::Equal => { continue; }
                                        Ordering::Greater => { break; }
                                    },
                                    _ => {
                                        self.tcx.sess.span_err(item_sp,
                                            "Invalid stability or deprecation version found");
                                        // act like it isn't less because the question is now
                                        // nonsensical, and this makes us not do anything else
                                        // interesting.
                                        break;
                                    }
                                }
                            }
                            is_less
                        },
                        _ => false,
                    };

                    if deprecated_predates_stable {
                        self.tcx.sess.span_err(item_sp,
                            "An API can't be stabilized after it is deprecated");
                    }

                    let def_id = self.tcx.map.local_def_id(id);
                    self.index.map.insert(def_id, Some(stab));

                    // Don't inherit #[stable(feature = "rust1", since = "1.0.0")]
                    if !stab.level.is_stable() {
                        let parent = replace(&mut self.parent, Some(stab));
                        f(self);
                        self.parent = parent;
                    } else {
                        f(self);
                    }
                }
                None => {
                    debug!("annotate: not found, use_parent = {:?}, parent = {:?}",
                           use_parent, self.parent);
                    if use_parent {
                        if let Some(stab) = self.parent {
                            let def_id = self.tcx.map.local_def_id(id);
                            self.index.map.insert(def_id, Some(stab));
                        } else if self.index.staged_api[&LOCAL_CRATE] && required
                            && self.export_map.contains(&id)
                            && !self.tcx.sess.opts.test {
                                self.tcx.sess.span_err(item_sp,
                                                       "This node does not \
                                                        have a stability attribute");
                            }
                    }
                    f(self);
                }
            }
        } else {
            // Emit warnings for non-staged-api crates. These should be errors.
            for attr in attrs {
                let tag = attr.name();
                if tag == "unstable" || tag == "stable" || tag == "deprecated" {
                    attr::mark_used(attr);
                    self.tcx.sess.span_err(attr.span(),
                                       "stability attributes may not be used outside \
                                        of the standard library");
                }
            }
            f(self);
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for Annotator<'a, 'tcx> {
    fn visit_item(&mut self, i: &Item) {
        // FIXME (#18969): the following is a hack around the fact
        // that we cannot currently annotate the stability of
        // `deriving`.  Basically, we do *not* allow stability
        // inheritance on trait implementations, so that derived
        // implementations appear to be unannotated. This then allows
        // derived implementations to be automatically tagged with the
        // stability of the trait. This is WRONG, but expedient to get
        // libstd stabilized for the 1.0 release.
        let use_parent = match i.node {
            hir::ItemImpl(_, _, _, Some(_), _, _) => false,
            _ => true,
        };

        // In case of a `pub use <mod>;`, we should not error since the stability
        // is inherited from the module itself
        let required = match i.node {
            hir::ItemUse(_) => i.vis != hir::Public,
            _ => true
        };

        self.annotate(i.id, use_parent, &i.attrs, i.span,
                      |v| visit::walk_item(v, i), required);

        if let hir::ItemStruct(ref sd, _) = i.node {
            if !sd.is_struct() {
                self.annotate(sd.id(), true, &i.attrs, i.span, |_| {}, true)
            }
        }
    }

    fn visit_fn(&mut self, _: FnKind<'v>, _: &'v FnDecl,
                _: &'v Block, _: Span, _: NodeId) {
        // Items defined in a function body have no reason to have
        // a stability attribute, so we don't recurse.
    }

    fn visit_trait_item(&mut self, ti: &hir::TraitItem) {
        self.annotate(ti.id, true, &ti.attrs, ti.span,
                      |v| visit::walk_trait_item(v, ti), true);
    }

    fn visit_impl_item(&mut self, ii: &hir::ImplItem) {
        self.annotate(ii.id, true, &ii.attrs, ii.span,
                      |v| visit::walk_impl_item(v, ii), true);
    }

    fn visit_variant(&mut self, var: &Variant, g: &'v Generics, item_id: NodeId) {
        self.annotate(var.node.data.id(), true, &var.node.attrs, var.span,
                      |v| visit::walk_variant(v, var, g, item_id), true)
    }

    fn visit_struct_field(&mut self, s: &StructField) {
        self.annotate(s.node.id, true, &s.node.attrs, s.span,
                      |v| visit::walk_struct_field(v, s), true);
    }

    fn visit_foreign_item(&mut self, i: &hir::ForeignItem) {
        self.annotate(i.id, true, &i.attrs, i.span, |_| {}, true);
    }
}

impl<'tcx> Index<'tcx> {
    /// Construct the stability index for a crate being compiled.
    pub fn build(&mut self, tcx: &ty::ctxt<'tcx>, krate: &Crate, export_map: &PublicItems) {
        let mut annotator = Annotator {
            tcx: tcx,
            index: self,
            parent: None,
            export_map: export_map,
        };
        annotator.annotate(ast::CRATE_NODE_ID, true, &krate.attrs, krate.span,
                           |v| visit::walk_crate(v, krate), true);
    }

    pub fn new(krate: &Crate) -> Index {
        let mut is_staged_api = false;
        for attr in &krate.attrs {
            if &attr.name()[..] == "staged_api" {
                match attr.node.value.node {
                    ast::MetaWord(_) => {
                        attr::mark_used(attr);
                        is_staged_api = true;
                    }
                    _ => (/*pass*/)
                }
            }
        }
        let mut staged_api = FnvHashMap();
        staged_api.insert(LOCAL_CRATE, is_staged_api);
        Index {
            staged_api: staged_api,
            map: DefIdMap(),
        }
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors. Returns a list of all
/// features used.
pub fn check_unstable_api_usage(tcx: &ty::ctxt)
                                -> FnvHashMap<InternedString, StabilityLevel> {
    let ref active_lib_features = tcx.sess.features.borrow().declared_lib_features;

    // Put the active features into a map for quick lookup
    let active_features = active_lib_features.iter().map(|&(ref s, _)| s.clone()).collect();

    let mut checker = Checker {
        tcx: tcx,
        active_features: active_features,
        used_features: FnvHashMap(),
        in_skip_block: 0,
    };

    let krate = tcx.map.krate();
    visit::walk_crate(&mut checker, krate);

    let used_features = checker.used_features;
    return used_features;
}

struct Checker<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    active_features: FnvHashSet<InternedString>,
    used_features: FnvHashMap<InternedString, StabilityLevel>,
    // Within a block where feature gate checking can be skipped.
    in_skip_block: u32,
}

impl<'a, 'tcx> Checker<'a, 'tcx> {
    fn check(&mut self, id: DefId, span: Span, stab: &Option<&Stability>) {
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
                self.used_features.insert(feature.clone(), Unstable);

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
                self.used_features.insert(feature.clone(), StabilityLevel::from_attr_level(level));

                // Stable APIs are always ok to call and deprecated APIs are
                // handled by a lint.
            }
            None => {
                // This is an 'unmarked' API, which should not exist
                // in the standard library.
                if self.tcx.sess.features.borrow().unmarked_api {
                    self.tcx.sess.span_warn(span, "use of unmarked library feature");
                    self.tcx.sess.span_note(span, "this is either a bug in the library you are \
                                                   using or a bug in the compiler - please \
                                                   report it in both places");
                } else {
                    self.tcx.sess.span_err(span, "use of unmarked library feature");
                    self.tcx.sess.span_note(span, "this is either a bug in the library you are \
                                                   using or a bug in the compiler - please \
                                                   report it in both places");
                    self.tcx.sess.span_note(span, "use #![feature(unmarked_api)] in the \
                                                   crate attributes to override this");
                }
            }
        }
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for Checker<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        // When compiling with --test we don't enforce stability on the
        // compiler-generated test module, demarcated with `DUMMY_SP` plus the
        // name `__test`
        if item.span == DUMMY_SP && item.name.as_str() == "__test" { return }

        check_item(self.tcx, item, true,
                   &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_item(self, item);
    }

    fn visit_expr(&mut self, ex: &hir::Expr) {
        check_expr(self.tcx, ex,
                   &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_expr(self, ex);
    }

    fn visit_path(&mut self, path: &hir::Path, id: ast::NodeId) {
        check_path(self.tcx, path, id,
                   &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_path(self, path)
    }

    fn visit_path_list_item(&mut self, prefix: &hir::Path, item: &hir::PathListItem) {
        check_path_list_item(self.tcx, item,
                   &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_path_list_item(self, prefix, item)
    }

    fn visit_pat(&mut self, pat: &hir::Pat) {
        check_pat(self.tcx, pat,
                  &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_pat(self, pat)
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
        visit::walk_block(self, b);
        self.in_skip_block = old_skip_count;
    }
}

/// Helper for discovering nodes to check for stability
pub fn check_item(tcx: &ty::ctxt, item: &hir::Item, warn_about_defns: bool,
                  cb: &mut FnMut(DefId, Span, &Option<&Stability>)) {
    match item.node {
        hir::ItemExternCrate(_) => {
            // compiler-generated `extern crate` items have a dummy span.
            if item.span == DUMMY_SP { return }

            let cnum = match tcx.sess.cstore.find_extern_mod_stmt_cnum(item.id) {
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
            let trait_did = tcx.def_map.borrow().get(&t.ref_id).unwrap().def_id();
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
pub fn check_expr(tcx: &ty::ctxt, e: &hir::Expr,
                  cb: &mut FnMut(DefId, Span, &Option<&Stability>)) {
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
                _ => tcx.sess.span_bug(e.span,
                                       "stability::check_expr: named field access on non-struct")
            }
        }
        hir::ExprTupField(ref base_e, ref field) => {
            span = field.span;
            match tcx.expr_ty_adjusted(base_e).sty {
                ty::TyStruct(def, _) => def.struct_variant().fields[field.node].did,
                ty::TyTuple(..) => return,
                _ => tcx.sess.span_bug(e.span,
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
                    tcx.sess.span_bug(e.span,
                                      &format!("stability::check_expr: struct construction \
                                                of non-struct, type {:?}",
                                               type_));
                }
            }
        }
        _ => return
    };

    maybe_do_stability_check(tcx, id, span, cb);
}

pub fn check_path(tcx: &ty::ctxt, path: &hir::Path, id: ast::NodeId,
                  cb: &mut FnMut(DefId, Span, &Option<&Stability>)) {
    match tcx.def_map.borrow().get(&id).map(|d| d.full_def()) {
        Some(def::DefPrimTy(..)) => {}
        Some(def::DefSelfTy(..)) => {}
        Some(def) => {
            maybe_do_stability_check(tcx, def.def_id(), path.span, cb);
        }
        None => {}
    }
}

pub fn check_path_list_item(tcx: &ty::ctxt, item: &hir::PathListItem,
                  cb: &mut FnMut(DefId, Span, &Option<&Stability>)) {
    match tcx.def_map.borrow().get(&item.node.id()).map(|d| d.full_def()) {
        Some(def::DefPrimTy(..)) => {}
        Some(def) => {
            maybe_do_stability_check(tcx, def.def_id(), item.span, cb);
        }
        None => {}
    }
}

pub fn check_pat(tcx: &ty::ctxt, pat: &hir::Pat,
                 cb: &mut FnMut(DefId, Span, &Option<&Stability>)) {
    debug!("check_pat(pat = {:?})", pat);
    if is_internal(tcx, pat.span) { return; }

    let v = match tcx.pat_ty_opt(pat) {
        Some(&ty::TyS { sty: ty::TyStruct(def, _), .. }) => def.struct_variant(),
        Some(_) | None => return,
    };
    match pat.node {
        // Foo(a, b, c)
        hir::PatEnum(_, Some(ref pat_fields)) => {
            for (field, struct_field) in pat_fields.iter().zip(&v.fields) {
                // a .. pattern is fine, but anything positional is
                // not.
                if let hir::PatWild(hir::PatWildMulti) = field.node {
                    continue
                }
                maybe_do_stability_check(tcx, struct_field.did, field.span, cb)
            }
        }
        // Foo { a, b, c }
        hir::PatStruct(_, ref pat_fields, _) => {
            for field in pat_fields {
                let did = v.field_named(field.node.name).did;
                maybe_do_stability_check(tcx, did, field.span, cb);
            }
        }
        // everything else is fine.
        _ => {}
    }
}

fn maybe_do_stability_check(tcx: &ty::ctxt, id: DefId, span: Span,
                            cb: &mut FnMut(DefId, Span, &Option<&Stability>)) {
    if !is_staged_api(tcx, id) {
        debug!("maybe_do_stability_check: \
                skipping id={:?} since it is not staged_api", id);
        return;
    }
    if is_internal(tcx, span) {
        debug!("maybe_do_stability_check: \
                skipping span={:?} since it is internal", span);
        return;
    }
    let ref stability = lookup(tcx, id);
    debug!("maybe_do_stability_check: \
            inspecting id={:?} span={:?} of stability={:?}", id, span, stability);
    cb(id, span, stability);
}

fn is_internal(tcx: &ty::ctxt, span: Span) -> bool {
    tcx.sess.codemap().span_allows_unstable(span)
}

fn is_staged_api(tcx: &ty::ctxt, id: DefId) -> bool {
    match tcx.trait_item_of_item(id) {
        Some(ty::MethodTraitItemId(trait_method_id))
            if trait_method_id != id => {
                is_staged_api(tcx, trait_method_id)
            }
        _ => {
            *tcx.stability.borrow_mut().staged_api.entry(id.krate).or_insert_with(
                || csearch::is_staged_api(&tcx.sess.cstore, id.krate))
        }
    }
}

/// Lookup the stability for a node, loading external crate
/// metadata as necessary.
pub fn lookup<'tcx>(tcx: &ty::ctxt<'tcx>, id: DefId) -> Option<&'tcx Stability> {
    if let Some(st) = tcx.stability.borrow().map.get(&id) {
        return *st;
    }

    let st = lookup_uncached(tcx, id);
    tcx.stability.borrow_mut().map.insert(id, st);
    st
}

fn lookup_uncached<'tcx>(tcx: &ty::ctxt<'tcx>, id: DefId) -> Option<&'tcx Stability> {
    debug!("lookup(id={:?})", id);

    // is this definition the implementation of a trait method?
    match tcx.trait_item_of_item(id) {
        Some(ty::MethodTraitItemId(trait_method_id)) if trait_method_id != id => {
            debug!("lookup: trait_method_id={:?}", trait_method_id);
            return lookup(tcx, trait_method_id)
        }
        _ => {}
    }

    let item_stab = if id.is_local() {
        None // The stability cache is filled partially lazily
    } else {
        csearch::get_stability(&tcx.sess.cstore, id).map(|st| tcx.intern_stability(st))
    };

    item_stab.or_else(|| {
        if tcx.is_impl(id) {
            if let Some(trait_id) = tcx.trait_id_of_impl(id) {
                // FIXME (#18969): for the time being, simply use the
                // stability of the trait to determine the stability of any
                // unmarked impls for it. See FIXME above for more details.

                debug!("lookup: trait_id={:?}", trait_id);
                return lookup(tcx, trait_id);
            }
        }
        None
    })
}

/// Given the list of enabled features that were not language features (i.e. that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features(sess: &Session,
                                       lib_features_used: &FnvHashMap<InternedString,
                                                                      StabilityLevel>) {
    let ref declared_lib_features = sess.features.borrow().declared_lib_features;
    let mut remaining_lib_features: FnvHashMap<InternedString, Span>
        = declared_lib_features.clone().into_iter().collect();

    let stable_msg = "this feature is stable. attribute no longer needed";

    for &span in &sess.features.borrow().declared_stable_lang_features {
        sess.add_lint(lint::builtin::STABLE_FEATURES,
                      ast::CRATE_NODE_ID,
                      span,
                      stable_msg.to_string());
    }

    for (used_lib_feature, level) in lib_features_used {
        match remaining_lib_features.remove(used_lib_feature) {
            Some(span) => {
                if *level == Stable {
                    sess.add_lint(lint::builtin::STABLE_FEATURES,
                                  ast::CRATE_NODE_ID,
                                  span,
                                  stable_msg.to_string());
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
