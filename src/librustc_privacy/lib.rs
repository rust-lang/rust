// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "rustc_privacy"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![cfg_attr(stage0, staged_api)]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;

extern crate rustc;
extern crate rustc_front;

use self::PrivacyResult::*;
use self::FieldName::*;

use std::cmp;
use std::mem::replace;

use rustc_front::hir;
use rustc_front::intravisit::{self, Visitor};

use rustc::lint;
use rustc::middle::def;
use rustc::middle::def_id::DefId;
use rustc::middle::privacy::{AccessLevel, AccessLevels};
use rustc::middle::privacy::ImportUse::*;
use rustc::middle::privacy::LastPrivate::*;
use rustc::middle::privacy::PrivateDep::*;
use rustc::middle::privacy::ExternalExports;
use rustc::middle::ty;
use rustc::util::nodemap::{NodeMap, NodeSet};
use rustc::front::map as ast_map;

use syntax::ast;
use syntax::codemap::Span;

pub mod diagnostics;

type Context<'a, 'tcx> = (&'a ty::MethodMap<'tcx>, &'a def::ExportMap);

/// Result of a checking operation - None => no errors were found. Some => an
/// error and contains the span and message for reporting that error and
/// optionally the same for a note about the error.
type CheckResult = Option<(Span, String, Option<(Span, String)>)>;

////////////////////////////////////////////////////////////////////////////////
/// The parent visitor, used to determine what's the parent of what (node-wise)
////////////////////////////////////////////////////////////////////////////////

struct ParentVisitor<'a, 'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
    parents: NodeMap<ast::NodeId>,
    curparent: ast::NodeId,
}

impl<'a, 'tcx, 'v> Visitor<'v> for ParentVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.tcx.map.expect_item(item.id))
    }
    fn visit_item(&mut self, item: &hir::Item) {
        self.parents.insert(item.id, self.curparent);

        let prev = self.curparent;
        match item.node {
            hir::ItemMod(..) => { self.curparent = item.id; }
            // Enum variants are parented to the enum definition itself because
            // they inherit privacy
            hir::ItemEnum(ref def, _) => {
                for variant in &def.variants {
                    // The parent is considered the enclosing enum because the
                    // enum will dictate the privacy visibility of this variant
                    // instead.
                    self.parents.insert(variant.node.data.id(), item.id);
                }
            }

            // Trait methods are always considered "public", but if the trait is
            // private then we need some private item in the chain from the
            // method to the root. In this case, if the trait is private, then
            // parent all the methods to the trait to indicate that they're
            // private.
            hir::ItemTrait(_, _, _, ref trait_items) if item.vis != hir::Public => {
                for trait_item in trait_items {
                    self.parents.insert(trait_item.id, item.id);
                }
            }

            _ => {}
        }
        intravisit::walk_item(self, item);
        self.curparent = prev;
    }

    fn visit_foreign_item(&mut self, a: &hir::ForeignItem) {
        self.parents.insert(a.id, self.curparent);
        intravisit::walk_foreign_item(self, a);
    }

    fn visit_fn(&mut self, a: intravisit::FnKind<'v>, b: &'v hir::FnDecl,
                c: &'v hir::Block, d: Span, id: ast::NodeId) {
        // We already took care of some trait methods above, otherwise things
        // like impl methods and pub trait methods are parented to the
        // containing module, not the containing trait.
        if !self.parents.contains_key(&id) {
            self.parents.insert(id, self.curparent);
        }
        intravisit::walk_fn(self, a, b, c, d);
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem) {
        // visit_fn handles methods, but associated consts have to be handled
        // here.
        if !self.parents.contains_key(&ii.id) {
            self.parents.insert(ii.id, self.curparent);
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant_data(&mut self, s: &hir::VariantData, _: ast::Name,
                        _: &'v hir::Generics, item_id: ast::NodeId, _: Span) {
        // Struct constructors are parented to their struct definitions because
        // they essentially are the struct definitions.
        if !s.is_struct() {
            self.parents.insert(s.id(), item_id);
        }

        // While we have the id of the struct definition, go ahead and parent
        // all the fields.
        for field in s.fields() {
            self.parents.insert(field.node.id, self.curparent);
        }
        intravisit::walk_struct_def(self, s)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the ast
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    export_map: &'a def::ExportMap,

    // Accessibility levels for reachable nodes
    access_levels: AccessLevels,
    // Previous accessibility level, None means unreachable
    prev_level: Option<AccessLevel>,
    // Have something changed in the level map?
    changed: bool,
}

impl<'a, 'tcx> EmbargoVisitor<'a, 'tcx> {
    fn ty_level(&self, ty: &hir::Ty) -> Option<AccessLevel> {
        if let hir::TyPath(..) = ty.node {
            match self.tcx.def_map.borrow().get(&ty.id).unwrap().full_def() {
                def::DefPrimTy(..) | def::DefSelfTy(..) | def::DefTyParam(..) => {
                    Some(AccessLevel::Public)
                }
                def => {
                    if let Some(node_id) = self.tcx.map.as_local_node_id(def.def_id()) {
                        self.get(node_id)
                    } else {
                        Some(AccessLevel::Public)
                    }
                }
            }
        } else {
            Some(AccessLevel::Public)
        }
    }

    fn trait_level(&self, trait_ref: &hir::TraitRef) -> Option<AccessLevel> {
        let did = self.tcx.trait_ref_to_def_id(trait_ref);
        if let Some(node_id) = self.tcx.map.as_local_node_id(did) {
            self.get(node_id)
        } else {
            Some(AccessLevel::Public)
        }
    }

    fn get(&self, id: ast::NodeId) -> Option<AccessLevel> {
        self.access_levels.map.get(&id).cloned()
    }

    // Updates node level and returns the updated level
    fn update(&mut self, id: ast::NodeId, level: Option<AccessLevel>) -> Option<AccessLevel> {
        let old_level = self.get(id);
        // Accessibility levels can only grow
        if level > old_level {
            self.access_levels.map.insert(id, level.unwrap());
            self.changed = true;
            level
        } else {
            old_level
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for EmbargoVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &hir::Item) {
        let inherited_item_level = match item.node {
            // Impls inherit level from their types and traits
            hir::ItemImpl(_, _, _, None, ref ty, _) => {
                self.ty_level(&ty)
            }
            hir::ItemImpl(_, _, _, Some(ref trait_ref), ref ty, _) => {
                cmp::min(self.ty_level(&ty), self.trait_level(trait_ref))
            }
            hir::ItemDefaultImpl(_, ref trait_ref) => {
                self.trait_level(trait_ref)
            }
            // Foreign mods inherit level from parents
            hir::ItemForeignMod(..) => {
                self.prev_level
            }
            // Other `pub` items inherit levels from parents
            _ => {
                if item.vis == hir::Public { self.prev_level } else { None }
            }
        };

        // Update id of the item itself
        let item_level = self.update(item.id, inherited_item_level);

        // Update ids of nested things
        match item.node {
            hir::ItemEnum(ref def, _) => {
                for variant in &def.variants {
                    let variant_level = self.update(variant.node.data.id(), item_level);
                    for field in variant.node.data.fields() {
                        self.update(field.node.id, variant_level);
                    }
                }
            }
            hir::ItemImpl(_, _, _, None, _, ref impl_items) => {
                for impl_item in impl_items {
                    if impl_item.vis == hir::Public {
                        self.update(impl_item.id, item_level);
                    }
                }
            }
            hir::ItemImpl(_, _, _, Some(_), _, ref impl_items) => {
                for impl_item in impl_items {
                    self.update(impl_item.id, item_level);
                }
            }
            hir::ItemTrait(_, _, _, ref trait_items) => {
                for trait_item in trait_items {
                    self.update(trait_item.id, item_level);
                }
            }
            hir::ItemStruct(ref def, _) => {
                if !def.is_struct() {
                    self.update(def.id(), item_level);
                }
                for field in def.fields() {
                    if field.node.kind.visibility() == hir::Public {
                        self.update(field.node.id, item_level);
                    }
                }
            }
            hir::ItemForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if foreign_item.vis == hir::Public {
                        self.update(foreign_item.id, item_level);
                    }
                }
            }
            hir::ItemTy(ref ty, _) if item_level.is_some() => {
                if let hir::TyPath(..) = ty.node {
                    match self.tcx.def_map.borrow().get(&ty.id).unwrap().full_def() {
                        def::DefPrimTy(..) | def::DefSelfTy(..) | def::DefTyParam(..) => {},
                        def => {
                            if let Some(node_id) = self.tcx.map.as_local_node_id(def.def_id()) {
                                self.update(node_id, Some(AccessLevel::Reachable));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        let orig_level = self.prev_level;
        self.prev_level = item_level;

        intravisit::walk_item(self, item);

        self.prev_level = orig_level;
    }

    fn visit_block(&mut self, b: &'v hir::Block) {
        let orig_level = replace(&mut self.prev_level, None);

        // Blocks can have public items, for example impls, but they always
        // start as completely private regardless of publicity of a function,
        // constant, type, field, etc. in which this block resides
        intravisit::walk_block(self, b);

        self.prev_level = orig_level;
    }

    fn visit_mod(&mut self, m: &hir::Mod, _sp: Span, id: ast::NodeId) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_level.is_some() {
            for export in self.export_map.get(&id).expect("module isn't found in export map") {
                if let Some(node_id) = self.tcx.map.as_local_node_id(export.def_id) {
                    self.update(node_id, Some(AccessLevel::Exported));
                }
            }
        }

        intravisit::walk_mod(self, m);
    }

    fn visit_macro_def(&mut self, md: &'v hir::MacroDef) {
        self.update(md.id, Some(AccessLevel::Public));
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy visitor, where privacy checks take place (violations reported)
////////////////////////////////////////////////////////////////////////////////

struct PrivacyVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    curitem: ast::NodeId,
    in_foreign: bool,
    parents: NodeMap<ast::NodeId>,
    external_exports: ExternalExports,
}

#[derive(Debug)]
enum PrivacyResult {
    Allowable,
    ExternallyDenied,
    DisallowedBy(ast::NodeId),
}

enum FieldName {
    UnnamedField(usize), // index
    NamedField(ast::Name),
}

impl<'a, 'tcx> PrivacyVisitor<'a, 'tcx> {
    // used when debugging
    fn nodestr(&self, id: ast::NodeId) -> String {
        self.tcx.map.node_to_string(id).to_string()
    }

    // Determines whether the given definition is public from the point of view
    // of the current item.
    fn def_privacy(&self, did: DefId) -> PrivacyResult {
        let node_id = if let Some(node_id) = self.tcx.map.as_local_node_id(did) {
            node_id
        } else {
            if self.external_exports.contains(&did) {
                debug!("privacy - {:?} was externally exported", did);
                return Allowable;
            }
            debug!("privacy - is {:?} a public method", did);

            return match self.tcx.impl_or_trait_items.borrow().get(&did) {
                Some(&ty::ConstTraitItem(ref ac)) => {
                    debug!("privacy - it's a const: {:?}", *ac);
                    match ac.container {
                        ty::TraitContainer(id) => {
                            debug!("privacy - recursing on trait {:?}", id);
                            self.def_privacy(id)
                        }
                        ty::ImplContainer(id) => {
                            match self.tcx.impl_trait_ref(id) {
                                Some(t) => {
                                    debug!("privacy - impl of trait {:?}", id);
                                    self.def_privacy(t.def_id)
                                }
                                None => {
                                    debug!("privacy - found inherent \
                                            associated constant {:?}",
                                            ac.vis);
                                    if ac.vis == hir::Public {
                                        Allowable
                                    } else {
                                        ExternallyDenied
                                    }
                                }
                            }
                        }
                    }
                }
                Some(&ty::MethodTraitItem(ref meth)) => {
                    debug!("privacy - well at least it's a method: {:?}",
                           *meth);
                    match meth.container {
                        ty::TraitContainer(id) => {
                            debug!("privacy - recursing on trait {:?}", id);
                            self.def_privacy(id)
                        }
                        ty::ImplContainer(id) => {
                            match self.tcx.impl_trait_ref(id) {
                                Some(t) => {
                                    debug!("privacy - impl of trait {:?}", id);
                                    self.def_privacy(t.def_id)
                                }
                                None => {
                                    debug!("privacy - found a method {:?}",
                                            meth.vis);
                                    if meth.vis == hir::Public {
                                        Allowable
                                    } else {
                                        ExternallyDenied
                                    }
                                }
                            }
                        }
                    }
                }
                Some(&ty::TypeTraitItem(ref typedef)) => {
                    match typedef.container {
                        ty::TraitContainer(id) => {
                            debug!("privacy - recursing on trait {:?}", id);
                            self.def_privacy(id)
                        }
                        ty::ImplContainer(id) => {
                            match self.tcx.impl_trait_ref(id) {
                                Some(t) => {
                                    debug!("privacy - impl of trait {:?}", id);
                                    self.def_privacy(t.def_id)
                                }
                                None => {
                                    debug!("privacy - found a typedef {:?}",
                                            typedef.vis);
                                    if typedef.vis == hir::Public {
                                        Allowable
                                    } else {
                                        ExternallyDenied
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    debug!("privacy - nope, not even a method");
                    ExternallyDenied
                }
            };
        };

        debug!("privacy - local {} not public all the way down",
               self.tcx.map.node_to_string(node_id));
        // return quickly for things in the same module
        if self.parents.get(&node_id) == self.parents.get(&self.curitem) {
            debug!("privacy - same parent, we're done here");
            return Allowable;
        }

        // We now know that there is at least one private member between the
        // destination and the root.
        let mut closest_private_id = node_id;
        loop {
            debug!("privacy - examining {}", self.nodestr(closest_private_id));
            let vis = match self.tcx.map.find(closest_private_id) {
                // If this item is a method, then we know for sure that it's an
                // actual method and not a static method. The reason for this is
                // that these cases are only hit in the ExprMethodCall
                // expression, and ExprCall will have its path checked later
                // (the path of the trait/impl) if it's a static method.
                //
                // With this information, then we can completely ignore all
                // trait methods. The privacy violation would be if the trait
                // couldn't get imported, not if the method couldn't be used
                // (all trait methods are public).
                //
                // However, if this is an impl method, then we dictate this
                // decision solely based on the privacy of the method
                // invocation.
                // FIXME(#10573) is this the right behavior? Why not consider
                //               where the method was defined?
                Some(ast_map::NodeImplItem(ii)) => {
                    match ii.node {
                        hir::ImplItemKind::Const(..) |
                        hir::ImplItemKind::Method(..) => {
                            let imp = self.tcx.map
                                          .get_parent_did(closest_private_id);
                            match self.tcx.impl_trait_ref(imp) {
                                Some(..) => return Allowable,
                                _ if ii.vis == hir::Public => {
                                    return Allowable
                                }
                                _ => ii.vis
                            }
                        }
                        hir::ImplItemKind::Type(_) => return Allowable,
                    }
                }
                Some(ast_map::NodeTraitItem(_)) => {
                    return Allowable;
                }

                // This is not a method call, extract the visibility as one
                // would normally look at it
                Some(ast_map::NodeItem(it)) => it.vis,
                Some(ast_map::NodeForeignItem(_)) => {
                    self.tcx.map.get_foreign_vis(closest_private_id)
                }
                Some(ast_map::NodeVariant(..)) => {
                    hir::Public // need to move up a level (to the enum)
                }
                _ => hir::Public,
            };
            if vis != hir::Public { break }
            // if we've reached the root, then everything was allowable and this
            // access is public.
            if closest_private_id == ast::CRATE_NODE_ID { return Allowable }
            closest_private_id = *self.parents.get(&closest_private_id).unwrap();

            // If we reached the top, then we were public all the way down and
            // we can allow this access.
            if closest_private_id == ast::DUMMY_NODE_ID { return Allowable }
        }
        debug!("privacy - closest priv {}", self.nodestr(closest_private_id));
        if self.private_accessible(closest_private_id) {
            Allowable
        } else {
            DisallowedBy(closest_private_id)
        }
    }

    /// True if `id` is both local and private-accessible
    fn local_private_accessible(&self, did: DefId) -> bool {
        if let Some(node_id) = self.tcx.map.as_local_node_id(did) {
            self.private_accessible(node_id)
        } else {
            false
        }
    }

    /// For a local private node in the AST, this function will determine
    /// whether the node is accessible by the current module that iteration is
    /// inside.
    fn private_accessible(&self, id: ast::NodeId) -> bool {
        let parent = *self.parents.get(&id).unwrap();
        debug!("privacy - accessible parent {}", self.nodestr(parent));

        // After finding `did`'s closest private member, we roll ourselves back
        // to see if this private member's parent is anywhere in our ancestry.
        // By the privacy rules, we can access all of our ancestor's private
        // members, so that's why we test the parent, and not the did itself.
        let mut cur = self.curitem;
        loop {
            debug!("privacy - questioning {}, {}", self.nodestr(cur), cur);
            match cur {
                // If the relevant parent is in our history, then we're allowed
                // to look inside any of our ancestor's immediate private items,
                // so this access is valid.
                x if x == parent => return true,

                // If we've reached the root, then we couldn't access this item
                // in the first place
                ast::DUMMY_NODE_ID => return false,

                // Keep going up
                _ => {}
            }

            cur = *self.parents.get(&cur).unwrap();
        }
    }

    fn report_error(&self, result: CheckResult) -> bool {
        match result {
            None => true,
            Some((span, msg, note)) => {
                self.tcx.sess.span_err(span, &msg[..]);
                match note {
                    Some((span, msg)) => {
                        self.tcx.sess.span_note(span, &msg[..])
                    }
                    None => {},
                }
                false
            },
        }
    }

    /// Guarantee that a particular definition is public. Returns a CheckResult
    /// which contains any errors found. These can be reported using `report_error`.
    /// If the result is `None`, no errors were found.
    fn ensure_public(&self,
                     span: Span,
                     to_check: DefId,
                     source_did: Option<DefId>,
                     msg: &str)
                     -> CheckResult {
        debug!("ensure_public(span={:?}, to_check={:?}, source_did={:?}, msg={:?})",
               span, to_check, source_did, msg);
        let def_privacy = self.def_privacy(to_check);
        debug!("ensure_public: def_privacy={:?}", def_privacy);
        let id = match def_privacy {
            ExternallyDenied => {
                return Some((span, format!("{} is private", msg), None))
            }
            Allowable => return None,
            DisallowedBy(id) => id,
        };

        // If we're disallowed by a particular id, then we attempt to
        // give a nice error message to say why it was disallowed. It
        // was either because the item itself is private or because
        // its parent is private and its parent isn't in our
        // ancestry. (Both the item being checked and its parent must
        // be local.)
        let def_id = source_did.unwrap_or(to_check);
        let node_id = self.tcx.map.as_local_node_id(def_id);
        let (err_span, err_msg) = if Some(id) == node_id {
            return Some((span, format!("{} is private", msg), None));
        } else {
            (span, format!("{} is inaccessible", msg))
        };
        let item = match self.tcx.map.find(id) {
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    // If an impl disallowed this item, then this is resolve's
                    // way of saying that a struct/enum's static method was
                    // invoked, and the struct/enum itself is private. Crawl
                    // back up the chains to find the relevant struct/enum that
                    // was private.
                    hir::ItemImpl(_, _, _, _, ref ty, _) => {
                        match ty.node {
                            hir::TyPath(..) => {}
                            _ => return Some((err_span, err_msg, None)),
                        };
                        let def = self.tcx.def_map.borrow().get(&ty.id).unwrap().full_def();
                        let did = def.def_id();
                        let node_id = self.tcx.map.as_local_node_id(did).unwrap();
                        match self.tcx.map.get(node_id) {
                            ast_map::NodeItem(item) => item,
                            _ => self.tcx.sess.span_bug(item.span,
                                                        "path is not an item")
                        }
                    }
                    _ => item
                }
            }
            Some(..) | None => return Some((err_span, err_msg, None)),
        };
        let desc = match item.node {
            hir::ItemMod(..) => "module",
            hir::ItemTrait(..) => "trait",
            hir::ItemStruct(..) => "struct",
            hir::ItemEnum(..) => "enum",
            _ => return Some((err_span, err_msg, None))
        };
        let msg = format!("{} `{}` is private", desc, item.name);
        Some((err_span, err_msg, Some((span, msg))))
    }

    // Checks that a field is in scope.
    fn check_field(&mut self,
                   span: Span,
                   def: ty::AdtDef<'tcx>,
                   v: ty::VariantDef<'tcx>,
                   name: FieldName) {
        let field = match name {
            NamedField(f_name) => {
                debug!("privacy - check named field {} in struct {:?}", f_name, def);
                v.field_named(f_name)
            }
            UnnamedField(idx) => &v.fields[idx]
        };
        if field.vis == hir::Public || self.local_private_accessible(field.did) {
            return;
        }

        let struct_desc = match def.adt_kind() {
            ty::AdtKind::Struct =>
                format!("struct `{}`", self.tcx.item_path_str(def.did)),
            // struct variant fields have inherited visibility
            ty::AdtKind::Enum => return
        };
        let msg = match name {
            NamedField(name) => format!("field `{}` of {} is private",
                                        name, struct_desc),
            UnnamedField(idx) => format!("field #{} of {} is private",
                                         idx + 1, struct_desc),
        };
        span_err!(self.tcx.sess, span, E0451,
                  "{}", &msg[..]);
    }

    // Given the ID of a method, checks to ensure it's in scope.
    fn check_static_method(&mut self,
                           span: Span,
                           method_id: DefId,
                           name: ast::Name) {
        self.report_error(self.ensure_public(span,
                                             method_id,
                                             None,
                                             &format!("method `{}`",
                                                     name)));
    }

    // Checks that a path is in scope.
    fn check_path(&mut self, span: Span, path_id: ast::NodeId, last: ast::Name) {
        debug!("privacy - path {}", self.nodestr(path_id));
        let path_res = *self.tcx.def_map.borrow().get(&path_id).unwrap();
        let ck = |tyname: &str| {
            let ck_public = |def: DefId| {
                debug!("privacy - ck_public {:?}", def);
                let origdid = path_res.def_id();
                self.ensure_public(span,
                                   def,
                                   Some(origdid),
                                   &format!("{} `{}`", tyname, last))
            };

            match path_res.last_private {
                LastMod(AllPublic) => {},
                LastMod(DependsOn(def)) => {
                    self.report_error(ck_public(def));
                },
                LastImport { value_priv,
                             value_used: check_value,
                             type_priv,
                             type_used: check_type } => {
                    // This dance with found_error is because we don't want to
                    // report a privacy error twice for the same directive.
                    let found_error = match (type_priv, check_type) {
                        (Some(DependsOn(def)), Used) => {
                            !self.report_error(ck_public(def))
                        },
                        _ => false,
                    };
                    if !found_error {
                        match (value_priv, check_value) {
                            (Some(DependsOn(def)), Used) => {
                                self.report_error(ck_public(def));
                            },
                            _ => {},
                        }
                    }
                    // If an import is not used in either namespace, we still
                    // want to check that it could be legal. Therefore we check
                    // in both namespaces and only report an error if both would
                    // be illegal. We only report one error, even if it is
                    // illegal to import from both namespaces.
                    match (value_priv, check_value, type_priv, check_type) {
                        (Some(p), Unused, None, _) |
                        (None, _, Some(p), Unused) => {
                            let p = match p {
                                AllPublic => None,
                                DependsOn(def) => ck_public(def),
                            };
                            if p.is_some() {
                                self.report_error(p);
                            }
                        },
                        (Some(v), Unused, Some(t), Unused) => {
                            let v = match v {
                                AllPublic => None,
                                DependsOn(def) => ck_public(def),
                            };
                            let t = match t {
                                AllPublic => None,
                                DependsOn(def) => ck_public(def),
                            };
                            if let (Some(_), Some(t)) = (v, t) {
                                self.report_error(Some(t));
                            }
                        },
                        _ => {},
                    }
                },
            }
        };
        // FIXME(#12334) Imports can refer to definitions in both the type and
        // value namespaces. The privacy information is aware of this, but the
        // def map is not. Therefore the names we work out below will not always
        // be accurate and we can get slightly wonky error messages (but type
        // checking is always correct).
        match path_res.full_def() {
            def::DefFn(..) => ck("function"),
            def::DefStatic(..) => ck("static"),
            def::DefConst(..) => ck("const"),
            def::DefAssociatedConst(..) => ck("associated const"),
            def::DefVariant(..) => ck("variant"),
            def::DefTy(_, false) => ck("type"),
            def::DefTy(_, true) => ck("enum"),
            def::DefTrait(..) => ck("trait"),
            def::DefStruct(..) => ck("struct"),
            def::DefMethod(..) => ck("method"),
            def::DefMod(..) => ck("module"),
            _ => {}
        }
    }

    // Checks that a method is in scope.
    fn check_method(&mut self, span: Span, method_def_id: DefId,
                    name: ast::Name) {
        match self.tcx.impl_or_trait_item(method_def_id).container() {
            ty::ImplContainer(_) => {
                self.check_static_method(span, method_def_id, name)
            }
            // Trait methods are always all public. The only controlling factor
            // is whether the trait itself is accessible or not.
            ty::TraitContainer(trait_def_id) => {
                self.report_error(self.ensure_public(span, trait_def_id,
                                                     None, "source trait"));
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for PrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &hir::Item) {
        let orig_curitem = replace(&mut self.curitem, item.id);
        intravisit::walk_item(self, item);
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprField(ref base, name) => {
                if let ty::TyStruct(def, _) = self.tcx.expr_ty_adjusted(&**base).sty {
                    self.check_field(expr.span,
                                     def,
                                     def.struct_variant(),
                                     NamedField(name.node));
                }
            }
            hir::ExprTupField(ref base, idx) => {
                if let ty::TyStruct(def, _) = self.tcx.expr_ty_adjusted(&**base).sty {
                    self.check_field(expr.span,
                                     def,
                                     def.struct_variant(),
                                     UnnamedField(idx.node));
                }
            }
            hir::ExprMethodCall(name, _, _) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let method = self.tcx.tables.borrow().method_map[&method_call];
                debug!("(privacy checking) checking impl method");
                self.check_method(expr.span, method.def_id, name.node);
            }
            hir::ExprStruct(..) => {
                let adt = self.tcx.expr_ty(expr).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(self.tcx.resolve_expr(expr));
                // RFC 736: ensure all unmentioned fields are visible.
                // Rather than computing the set of unmentioned fields
                // (i.e. `all_fields - fields`), just check them all.
                for field in &variant.fields {
                    self.check_field(expr.span, adt, variant, NamedField(field.name));
                }
            }
            hir::ExprPath(..) => {

                if let def::DefStruct(_) = self.tcx.resolve_expr(expr) {
                    let expr_ty = self.tcx.expr_ty(expr);
                    let def = match expr_ty.sty {
                        ty::TyBareFn(_, &ty::BareFnTy { sig: ty::Binder(ty::FnSig {
                            output: ty::FnConverging(ty), ..
                        }), ..}) => ty,
                        _ => expr_ty
                    }.ty_adt_def().unwrap();
                    let any_priv = def.struct_variant().fields.iter().any(|f| {
                        f.vis != hir::Public && !self.local_private_accessible(f.did)
                    });
                    if any_priv {
                        span_err!(self.tcx.sess, expr.span, E0450,
                                  "cannot invoke tuple struct constructor with private \
                                   fields");
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pattern: &hir::Pat) {
        // Foreign functions do not have their patterns mapped in the def_map,
        // and there's nothing really relevant there anyway, so don't bother
        // checking privacy. If you can name the type then you can pass it to an
        // external C function anyway.
        if self.in_foreign { return }

        match pattern.node {
            hir::PatStruct(_, ref fields, _) => {
                let adt = self.tcx.pat_ty(pattern).ty_adt_def().unwrap();
                let def = self.tcx.def_map.borrow().get(&pattern.id).unwrap().full_def();
                let variant = adt.variant_of_def(def);
                for field in fields {
                    self.check_field(pattern.span, adt, variant,
                                     NamedField(field.node.name));
                }
            }

            // Patterns which bind no fields are allowable (the path is check
            // elsewhere).
            hir::PatEnum(_, Some(ref fields)) => {
                match self.tcx.pat_ty(pattern).sty {
                    ty::TyStruct(def, _) => {
                        for (i, field) in fields.iter().enumerate() {
                            if let hir::PatWild = field.node {
                                continue
                            }
                            self.check_field(field.span,
                                             def,
                                             def.struct_variant(),
                                             UnnamedField(i));
                        }
                    }
                    ty::TyEnum(..) => {
                        // enum fields have no privacy at this time
                    }
                    _ => {}
                }

            }
            _ => {}
        }

        intravisit::walk_pat(self, pattern);
    }

    fn visit_foreign_item(&mut self, fi: &hir::ForeignItem) {
        self.in_foreign = true;
        intravisit::walk_foreign_item(self, fi);
        self.in_foreign = false;
    }

    fn visit_path(&mut self, path: &hir::Path, id: ast::NodeId) {
        if !path.segments.is_empty() {
            self.check_path(path.span, id, path.segments.last().unwrap().identifier.name);
            intravisit::walk_path(self, path);
        }
    }

    fn visit_path_list_item(&mut self, prefix: &hir::Path, item: &hir::PathListItem) {
        let name = if let hir::PathListIdent { name, .. } = item.node {
            name
        } else if !prefix.segments.is_empty() {
            prefix.segments.last().unwrap().identifier.name
        } else {
            self.tcx.sess.bug("`self` import in an import list with empty prefix");
        };
        self.check_path(item.span, item.node.id(), name);
        intravisit::walk_path_list_item(self, prefix, item);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy sanity check visitor, ensures unnecessary visibility isn't here
////////////////////////////////////////////////////////////////////////////////

struct SanePrivacyVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    in_block: bool,
}

impl<'a, 'tcx, 'v> Visitor<'v> for SanePrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &hir::Item) {
        self.check_sane_privacy(item);
        if self.in_block {
            self.check_all_inherited(item);
        }

        let orig_in_block = self.in_block;

        // Modules turn privacy back on, otherwise we inherit
        self.in_block = if let hir::ItemMod(..) = item.node { false } else { orig_in_block };

        intravisit::walk_item(self, item);
        self.in_block = orig_in_block;
    }

    fn visit_block(&mut self, b: &'v hir::Block) {
        let orig_in_block = replace(&mut self.in_block, true);
        intravisit::walk_block(self, b);
        self.in_block = orig_in_block;
    }
}

impl<'a, 'tcx> SanePrivacyVisitor<'a, 'tcx> {
    /// Validates all of the visibility qualifiers placed on the item given. This
    /// ensures that there are no extraneous qualifiers that don't actually do
    /// anything. In theory these qualifiers wouldn't parse, but that may happen
    /// later on down the road...
    fn check_sane_privacy(&self, item: &hir::Item) {
        let check_inherited = |sp, vis, note: &str| {
            if vis != hir::Inherited {
                span_err!(self.tcx.sess, sp, E0449, "unnecessary visibility qualifier");
                if !note.is_empty() {
                    self.tcx.sess.span_note(sp, note);
                }
            }
        };

        match item.node {
            // implementations of traits don't need visibility qualifiers because
            // that's controlled by having the trait in scope.
            hir::ItemImpl(_, _, _, Some(..), _, ref impl_items) => {
                check_inherited(item.span, item.vis,
                                "visibility qualifiers have no effect on trait impls");
                for impl_item in impl_items {
                    check_inherited(impl_item.span, impl_item.vis, "");
                }
            }
            hir::ItemImpl(_, _, _, None, _, _) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual methods instead");
            }
            hir::ItemDefaultImpl(..) => {
                check_inherited(item.span, item.vis,
                                "visibility qualifiers have no effect on trait impls");
            }
            hir::ItemForeignMod(..) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual functions instead");
            }
            hir::ItemStruct(..) | hir::ItemEnum(..) | hir::ItemTrait(..) |
            hir::ItemConst(..) | hir::ItemStatic(..) | hir::ItemFn(..) |
            hir::ItemMod(..) | hir::ItemExternCrate(..) |
            hir::ItemUse(..) | hir::ItemTy(..) => {}
        }
    }

    /// When inside of something like a function or a method, visibility has no
    /// control over anything so this forbids any mention of any visibility
    fn check_all_inherited(&self, item: &hir::Item) {
        let check_inherited = |sp, vis| {
            if vis != hir::Inherited {
                span_err!(self.tcx.sess, sp, E0447,
                          "visibility has no effect inside functions or block expressions");
            }
        };

        check_inherited(item.span, item.vis);
        match item.node {
            hir::ItemImpl(_, _, _, _, _, ref impl_items) => {
                for impl_item in impl_items {
                    check_inherited(impl_item.span, impl_item.vis);
                }
            }
            hir::ItemForeignMod(ref fm) => {
                for fi in &fm.items {
                    check_inherited(fi.span, fi.vis);
                }
            }
            hir::ItemStruct(ref vdata, _) => {
                for f in vdata.fields() {
                    check_inherited(f.span, f.node.kind.visibility());
                }
            }
            hir::ItemDefaultImpl(..) | hir::ItemEnum(..) | hir::ItemTrait(..) |
            hir::ItemConst(..) | hir::ItemStatic(..) | hir::ItemFn(..) |
            hir::ItemMod(..) | hir::ItemExternCrate(..) |
            hir::ItemUse(..) | hir::ItemTy(..) => {}
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Obsolete visitors for checking for private items in public interfaces.
/// These visitors are supposed to be kept in frozen state and produce an
/// "old error node set". For backward compatibility the new visitor reports
/// warnings instead of hard errors when the erroneous node is not in this old set.
///////////////////////////////////////////////////////////////////////////////

struct ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    access_levels: &'a AccessLevels,
    in_variant: bool,
    // set of errors produced by this obsolete visitor
    old_error_set: NodeSet,
}

struct ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b: 'a, 'tcx: 'b> {
    inner: &'a ObsoleteVisiblePrivateTypesVisitor<'b, 'tcx>,
    /// whether the type refers to private types.
    contains_private: bool,
    /// whether we've recurred at all (i.e. if we're pointing at the
    /// first type on which visit_ty was called).
    at_outer_type: bool,
    // whether that first type is a public path.
    outer_type_is_public_path: bool,
}

impl<'a, 'tcx> ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    fn path_is_private_type(&self, path_id: ast::NodeId) -> bool {
        let did = match self.tcx.def_map.borrow().get(&path_id).map(|d| d.full_def()) {
            // `int` etc. (None doesn't seem to occur.)
            None | Some(def::DefPrimTy(..)) | Some(def::DefSelfTy(..)) => return false,
            Some(def) => def.def_id(),
        };

        // A path can only be private if:
        // it's in this crate...
        if let Some(node_id) = self.tcx.map.as_local_node_id(did) {
            // .. and it corresponds to a private type in the AST (this returns
            // None for type parameters)
            match self.tcx.map.find(node_id) {
                Some(ast_map::NodeItem(ref item)) => item.vis != hir::Public,
                Some(_) | None => false,
            }
        } else {
            return false
        }
    }

    fn trait_is_public(&self, trait_id: ast::NodeId) -> bool {
        // FIXME: this would preferably be using `exported_items`, but all
        // traits are exported currently (see `EmbargoVisitor.exported_trait`)
        self.access_levels.is_public(trait_id)
    }

    fn check_ty_param_bound(&mut self,
                            ty_param_bound: &hir::TyParamBound) {
        if let hir::TraitTyParamBound(ref trait_ref, _) = *ty_param_bound {
            if self.path_is_private_type(trait_ref.trait_ref.ref_id) {
                self.old_error_set.insert(trait_ref.trait_ref.ref_id);
            }
        }
    }

    fn item_is_public(&self, id: &ast::NodeId, vis: hir::Visibility) -> bool {
        self.access_levels.is_reachable(*id) || vis == hir::Public
    }
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b, 'tcx> {
    fn visit_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyPath(..) = ty.node {
            if self.inner.path_is_private_type(ty.id) {
                self.contains_private = true;
                // found what we're looking for so let's stop
                // working.
                return
            } else if self.at_outer_type {
                self.outer_type_is_public_path = true;
            }
        }
        self.at_outer_type = false;
        intravisit::walk_ty(self, ty)
    }

    // don't want to recurse into [, .. expr]
    fn visit_expr(&mut self, _: &hir::Expr) {}
}

impl<'a, 'tcx, 'v> Visitor<'v> for ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &hir::Item) {
        match item.node {
            // contents of a private mod can be reexported, so we need
            // to check internals.
            hir::ItemMod(_) => {}

            // An `extern {}` doesn't introduce a new privacy
            // namespace (the contents have their own privacies).
            hir::ItemForeignMod(_) => {}

            hir::ItemTrait(_, _, ref bounds, _) => {
                if !self.trait_is_public(item.id) {
                    return
                }

                for bound in bounds.iter() {
                    self.check_ty_param_bound(bound)
                }
            }

            // impls need some special handling to try to offer useful
            // error messages without (too many) false positives
            // (i.e. we could just return here to not check them at
            // all, or some worse estimation of whether an impl is
            // publicly visible).
            hir::ItemImpl(_, _, ref g, ref trait_ref, ref self_, ref impl_items) => {
                // `impl [... for] Private` is never visible.
                let self_contains_private;
                // impl [... for] Public<...>, but not `impl [... for]
                // Vec<Public>` or `(Public,)` etc.
                let self_is_public_path;

                // check the properties of the Self type:
                {
                    let mut visitor = ObsoleteCheckTypeForPrivatenessVisitor {
                        inner: self,
                        contains_private: false,
                        at_outer_type: true,
                        outer_type_is_public_path: false,
                    };
                    visitor.visit_ty(&**self_);
                    self_contains_private = visitor.contains_private;
                    self_is_public_path = visitor.outer_type_is_public_path;
                }

                // miscellaneous info about the impl

                // `true` iff this is `impl Private for ...`.
                let not_private_trait =
                    trait_ref.as_ref().map_or(true, // no trait counts as public trait
                                              |tr| {
                        let did = self.tcx.trait_ref_to_def_id(tr);

                        if let Some(node_id) = self.tcx.map.as_local_node_id(did) {
                            self.trait_is_public(node_id)
                        } else {
                            true // external traits must be public
                        }
                    });

                // `true` iff this is a trait impl or at least one method is public.
                //
                // `impl Public { $( fn ...() {} )* }` is not visible.
                //
                // This is required over just using the methods' privacy
                // directly because we might have `impl<T: Foo<Private>> ...`,
                // and we shouldn't warn about the generics if all the methods
                // are private (because `T` won't be visible externally).
                let trait_or_some_public_method =
                    trait_ref.is_some() ||
                    impl_items.iter()
                              .any(|impl_item| {
                                  match impl_item.node {
                                      hir::ImplItemKind::Const(..) |
                                      hir::ImplItemKind::Method(..) => {
                                          self.access_levels.is_reachable(impl_item.id)
                                      }
                                      hir::ImplItemKind::Type(_) => false,
                                  }
                              });

                if !self_contains_private &&
                        not_private_trait &&
                        trait_or_some_public_method {

                    intravisit::walk_generics(self, g);

                    match *trait_ref {
                        None => {
                            for impl_item in impl_items {
                                // This is where we choose whether to walk down
                                // further into the impl to check its items. We
                                // should only walk into public items so that we
                                // don't erroneously report errors for private
                                // types in private items.
                                match impl_item.node {
                                    hir::ImplItemKind::Const(..) |
                                    hir::ImplItemKind::Method(..)
                                        if self.item_is_public(&impl_item.id, impl_item.vis) =>
                                    {
                                        intravisit::walk_impl_item(self, impl_item)
                                    }
                                    hir::ImplItemKind::Type(..) => {
                                        intravisit::walk_impl_item(self, impl_item)
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Some(ref tr) => {
                            // Any private types in a trait impl fall into three
                            // categories.
                            // 1. mentioned in the trait definition
                            // 2. mentioned in the type params/generics
                            // 3. mentioned in the associated types of the impl
                            //
                            // Those in 1. can only occur if the trait is in
                            // this crate and will've been warned about on the
                            // trait definition (there's no need to warn twice
                            // so we don't check the methods).
                            //
                            // Those in 2. are warned via walk_generics and this
                            // call here.
                            intravisit::walk_path(self, &tr.path);

                            // Those in 3. are warned with this call.
                            for impl_item in impl_items {
                                if let hir::ImplItemKind::Type(ref ty) = impl_item.node {
                                    self.visit_ty(ty);
                                }
                            }
                        }
                    }
                } else if trait_ref.is_none() && self_is_public_path {
                    // impl Public<Private> { ... }. Any public static
                    // methods will be visible as `Public::foo`.
                    let mut found_pub_static = false;
                    for impl_item in impl_items {
                        match impl_item.node {
                            hir::ImplItemKind::Const(..) => {
                                if self.item_is_public(&impl_item.id, impl_item.vis) {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                            }
                            hir::ImplItemKind::Method(ref sig, _) => {
                                if sig.explicit_self.node == hir::SelfStatic &&
                                      self.item_is_public(&impl_item.id, impl_item.vis) {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                            }
                            _ => {}
                        }
                    }
                    if found_pub_static {
                        intravisit::walk_generics(self, g)
                    }
                }
                return
            }

            // `type ... = ...;` can contain private types, because
            // we're introducing a new name.
            hir::ItemTy(..) => return,

            // not at all public, so we don't care
            _ if !self.item_is_public(&item.id, item.vis) => {
                return;
            }

            _ => {}
        }

        // We've carefully constructed it so that if we're here, then
        // any `visit_ty`'s will be called on things that are in
        // public signatures, i.e. things that we're interested in for
        // this visitor.
        debug!("VisiblePrivateTypesVisitor entering item {:?}", item);
        intravisit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &hir::Generics) {
        for ty_param in generics.ty_params.iter() {
            for bound in ty_param.bounds.iter() {
                self.check_ty_param_bound(bound)
            }
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &hir::WherePredicate::BoundPredicate(ref bound_pred) => {
                    for bound in bound_pred.bounds.iter() {
                        self.check_ty_param_bound(bound)
                    }
                }
                &hir::WherePredicate::RegionPredicate(_) => {}
                &hir::WherePredicate::EqPredicate(ref eq_pred) => {
                    self.visit_ty(&*eq_pred.ty);
                }
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &hir::ForeignItem) {
        if self.access_levels.is_reachable(item.id) {
            intravisit::walk_foreign_item(self, item)
        }
    }

    fn visit_ty(&mut self, t: &hir::Ty) {
        debug!("VisiblePrivateTypesVisitor checking ty {:?}", t);
        if let hir::TyPath(..) = t.node {
            if self.path_is_private_type(t.id) {
                self.old_error_set.insert(t.id);
            }
        }
        intravisit::walk_ty(self, t)
    }

    fn visit_variant(&mut self, v: &hir::Variant, g: &hir::Generics, item_id: ast::NodeId) {
        if self.access_levels.is_reachable(v.node.data.id()) {
            self.in_variant = true;
            intravisit::walk_variant(self, v, g, item_id);
            self.in_variant = false;
        }
    }

    fn visit_struct_field(&mut self, s: &hir::StructField) {
        let vis = match s.node.kind {
            hir::NamedField(_, vis) | hir::UnnamedField(vis) => vis
        };
        if vis == hir::Public || self.in_variant {
            intravisit::walk_struct_field(self, s);
        }
    }

    // we don't need to introspect into these at all: an
    // expression/block context can't possibly contain exported things.
    // (Making them no-ops stops us from traversing the whole AST without
    // having to be super careful about our `walk_...` calls above.)
    // FIXME(#29524): Unfortunately this ^^^ is not true, blocks can contain
    // exported items (e.g. impls) and actual code in rustc itself breaks
    // if we don't traverse blocks in `EmbargoVisitor`
    fn visit_block(&mut self, _: &hir::Block) {}
    fn visit_expr(&mut self, _: &hir::Expr) {}
}

///////////////////////////////////////////////////////////////////////////////
/// SearchInterfaceForPrivateItemsVisitor traverses an item's interface and
/// finds any private components in it.
/// PrivateItemsInPublicInterfacesVisitor ensures there are no private types
/// and traits in public interfaces.
///////////////////////////////////////////////////////////////////////////////

struct SearchInterfaceForPrivateItemsVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    // Do not report an error when a private type is found
    is_quiet: bool,
    // Is private component found?
    is_public: bool,
    old_error_set: &'a NodeSet,
}

impl<'a, 'tcx: 'a> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    // Check if the type alias contain private types when substituted
    fn is_public_type_alias(&self, item: &hir::Item, path: &hir::Path) -> bool {
        // Type alias is considered public if the aliased type is
        // public, even if the type alias itself is private. So, something
        // like `type A = u8; pub fn f() -> A {...}` doesn't cause an error.
        if let hir::ItemTy(ref ty, ref generics) = item.node {
            let mut check = SearchInterfaceForPrivateItemsVisitor {
                tcx: self.tcx, is_quiet: self.is_quiet,
                is_public: true, old_error_set: self.old_error_set,
            };
            check.visit_ty(ty);
            let provided_params = path.segments.last().unwrap().parameters.types().len();
            for ty_param in &generics.ty_params[provided_params..] {
                if let Some(ref default_ty) = ty_param.default {
                    check.visit_ty(default_ty);
                }
            }
            check.is_public
        } else {
            false
        }
    }
}

impl<'a, 'tcx: 'a, 'v> Visitor<'v> for SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: &hir::Ty) {
        if self.is_quiet && !self.is_public {
            // We are in quiet mode and a private type is already found, no need to proceed
            return
        }
        if let hir::TyPath(_, ref path) = ty.node {
            let def = self.tcx.def_map.borrow().get(&ty.id).unwrap().full_def();
            match def {
                def::DefPrimTy(..) | def::DefSelfTy(..) | def::DefTyParam(..) => {
                    // Public
                }
                def::DefAssociatedTy(..) if self.is_quiet => {
                    // Conservatively approximate the whole type alias as public without
                    // recursing into its components when determining impl publicity.
                    return
                }
                def::DefStruct(def_id) | def::DefTy(def_id, _) |
                def::DefTrait(def_id) | def::DefAssociatedTy(def_id, _) => {
                    // Non-local means public, local needs to be checked
                    if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
                        if let Some(ast_map::NodeItem(ref item)) = self.tcx.map.find(node_id) {
                            if item.vis != hir::Public && !self.is_public_type_alias(item, path) {
                                if !self.is_quiet {
                                    if self.old_error_set.contains(&ty.id) {
                                        span_err!(self.tcx.sess, ty.span, E0446,
                                                  "private type in public interface");
                                    } else {
                                        self.tcx.sess.add_lint (
                                            lint::builtin::PRIVATE_IN_PUBLIC,
                                            node_id,
                                            ty.span,
                                            "private type in public interface".to_string()
                                        );
                                    }
                                }
                                self.is_public = false;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        intravisit::walk_ty(self, ty);
    }

    fn visit_trait_ref(&mut self, trait_ref: &hir::TraitRef) {
        if self.is_quiet && !self.is_public {
            // We are in quiet mode and a private type is already found, no need to proceed
            return
        }
        // Non-local means public, local needs to be checked
        let def_id = self.tcx.trait_ref_to_def_id(trait_ref);
        if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
            if let Some(ast_map::NodeItem(ref item)) = self.tcx.map.find(node_id) {
                if item.vis != hir::Public {
                    if !self.is_quiet {
                        if self.old_error_set.contains(&trait_ref.ref_id) {
                            span_err!(self.tcx.sess, trait_ref.path.span, E0445,
                                      "private trait in public interface");
                        } else {
                            self.tcx.sess.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                                   node_id,
                                                   trait_ref.path.span,
                                                   "private trait in public interface".to_string());
                        }
                    }
                    self.is_public = false;
                }
            }
        }

        intravisit::walk_trait_ref(self, trait_ref);
    }

    // Don't recurse into function bodies
    fn visit_block(&mut self, _: &hir::Block) {}
    // Don't recurse into expressions in array sizes or const initializers
    fn visit_expr(&mut self, _: &hir::Expr) {}
    // Don't recurse into patterns in function arguments
    fn visit_pat(&mut self, _: &hir::Pat) {}
}

struct PrivateItemsInPublicInterfacesVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    old_error_set: &'a NodeSet,
}

impl<'a, 'tcx> PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    // A type is considered public if it doesn't contain any private components
    fn is_public_ty(&self, ty: &hir::Ty) -> bool {
        let mut check = SearchInterfaceForPrivateItemsVisitor {
            tcx: self.tcx, is_quiet: true, is_public: true, old_error_set: self.old_error_set
        };
        check.visit_ty(ty);
        check.is_public
    }

    // A trait is considered public if it doesn't contain any private components
    fn is_public_trait(&self, trait_ref: &hir::TraitRef) -> bool {
        let mut check = SearchInterfaceForPrivateItemsVisitor {
            tcx: self.tcx, is_quiet: true, is_public: true, old_error_set: self.old_error_set
        };
        check.visit_trait_ref(trait_ref);
        check.is_public
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        let mut check = SearchInterfaceForPrivateItemsVisitor {
            tcx: self.tcx, is_quiet: false, is_public: true, old_error_set: self.old_error_set
        };
        match item.node {
            // Crates are always public
            hir::ItemExternCrate(..) => {}
            // All nested items are checked by visit_item
            hir::ItemMod(..) => {}
            // Checked in resolve
            hir::ItemUse(..) => {}
            // Subitems of these items have inherited publicity
            hir::ItemConst(..) | hir::ItemStatic(..) | hir::ItemFn(..) |
            hir::ItemEnum(..) | hir::ItemTrait(..) | hir::ItemTy(..) => {
                if item.vis == hir::Public {
                    check.visit_item(item);
                }
            }
            // Subitems of foreign modules have their own publicity
            hir::ItemForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if foreign_item.vis == hir::Public {
                        check.visit_foreign_item(foreign_item);
                    }
                }
            }
            // Subitems of structs have their own publicity
            hir::ItemStruct(ref struct_def, ref generics) => {
                if item.vis == hir::Public {
                    check.visit_generics(generics);
                    for field in struct_def.fields() {
                        if field.node.kind.visibility() == hir::Public {
                            check.visit_struct_field(field);
                        }
                    }
                }
            }
            // The interface is empty
            hir::ItemDefaultImpl(..) => {}
            // An inherent impl is public when its type is public
            // Subitems of inherent impls have their own publicity
            hir::ItemImpl(_, _, ref generics, None, ref ty, ref impl_items) => {
                if self.is_public_ty(ty) {
                    check.visit_generics(generics);
                    for impl_item in impl_items {
                        if impl_item.vis == hir::Public {
                            check.visit_impl_item(impl_item);
                        }
                    }
                }
            }
            // A trait impl is public when both its type and its trait are public
            // Subitems of trait impls have inherited publicity
            hir::ItemImpl(_, _, ref generics, Some(ref trait_ref), ref ty, ref impl_items) => {
                if self.is_public_ty(ty) && self.is_public_trait(trait_ref) {
                    check.visit_generics(generics);
                    for impl_item in impl_items {
                        check.visit_impl_item(impl_item);
                    }
                }
            }
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   export_map: &def::ExportMap,
                   external_exports: ExternalExports)
                   -> AccessLevels {
    let krate = tcx.map.krate();

    // Sanity check to make sure that all privacy usage and controls are
    // reasonable.
    let mut visitor = SanePrivacyVisitor {
        tcx: tcx,
        in_block: false,
    };
    intravisit::walk_crate(&mut visitor, krate);

    // Figure out who everyone's parent is
    let mut visitor = ParentVisitor {
        tcx: tcx,
        parents: NodeMap(),
        curparent: ast::DUMMY_NODE_ID,
    };
    intravisit::walk_crate(&mut visitor, krate);

    // Use the parent map to check the privacy of everything
    let mut visitor = PrivacyVisitor {
        curitem: ast::DUMMY_NODE_ID,
        in_foreign: false,
        tcx: tcx,
        parents: visitor.parents,
        external_exports: external_exports,
    };
    intravisit::walk_crate(&mut visitor, krate);

    tcx.sess.abort_if_errors();

    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx: tcx,
        export_map: export_map,
        access_levels: Default::default(),
        prev_level: Some(AccessLevel::Public),
        changed: false,
    };
    loop {
        intravisit::walk_crate(&mut visitor, krate);
        if visitor.changed {
            visitor.changed = false;
        } else {
            break
        }
    }
    visitor.update(ast::CRATE_NODE_ID, Some(AccessLevel::Public));

    {
        let mut visitor = ObsoleteVisiblePrivateTypesVisitor {
            tcx: tcx,
            access_levels: &visitor.access_levels,
            in_variant: false,
            old_error_set: NodeSet(),
        };
        intravisit::walk_crate(&mut visitor, krate);

        // Check for private types and traits in public interfaces
        let mut visitor = PrivateItemsInPublicInterfacesVisitor {
            tcx: tcx,
            old_error_set: &visitor.old_error_set,
        };
        krate.visit_all_items(&mut visitor);
    }

    visitor.access_levels
}

__build_diagnostic_array! { librustc_privacy, DIAGNOSTICS }
