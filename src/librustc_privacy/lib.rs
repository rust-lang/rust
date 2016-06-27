// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_privacy"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

extern crate rustc;
#[macro_use] extern crate syntax;
extern crate syntax_pos;

use rustc::dep_graph::DepNode;
use rustc::hir::{self, PatKind};
use rustc::hir::def::{self, Def};
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor};
use rustc::hir::pat_util::EnumerateAndAdjustIterator;
use rustc::lint;
use rustc::middle::privacy::{AccessLevel, AccessLevels};
use rustc::ty::{self, TyCtxt};
use rustc::util::nodemap::NodeSet;
use syntax::ast;
use syntax_pos::Span;

use std::cmp;
use std::mem::replace;

pub mod diagnostics;

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the ast
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    export_map: &'a def::ExportMap,

    // Accessibility levels for reachable nodes
    access_levels: AccessLevels,
    // Previous accessibility level, None means unreachable
    prev_level: Option<AccessLevel>,
    // Have something changed in the level map?
    changed: bool,
}

struct ReachEverythingInTheInterfaceVisitor<'b, 'a: 'b, 'tcx: 'a> {
    ev: &'b mut EmbargoVisitor<'a, 'tcx>,
}

impl<'a, 'tcx> EmbargoVisitor<'a, 'tcx> {
    fn ty_level(&self, ty: &hir::Ty) -> Option<AccessLevel> {
        if let hir::TyPath(..) = ty.node {
            match self.tcx.expect_def(ty.id) {
                Def::PrimTy(..) | Def::SelfTy(..) | Def::TyParam(..) => {
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
        let did = self.tcx.expect_def(trait_ref.ref_id).def_id();
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

    fn reach<'b>(&'b mut self) -> ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
        ReachEverythingInTheInterfaceVisitor { ev: self }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for EmbargoVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        let tcx = self.tcx;
        self.visit_item(tcx.map.expect_item(item.id))
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

        // Update level of the item itself
        let item_level = self.update(item.id, inherited_item_level);

        // Update levels of nested things
        match item.node {
            hir::ItemEnum(ref def, _) => {
                for variant in &def.variants {
                    let variant_level = self.update(variant.node.data.id(), item_level);
                    for field in variant.node.data.fields() {
                        self.update(field.id, variant_level);
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
                    if field.vis == hir::Public {
                        self.update(field.id, item_level);
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
            _ => {}
        }

        // Mark all items in interfaces of reachable items as reachable
        match item.node {
            // The interface is empty
            hir::ItemExternCrate(..) => {}
            // All nested items are checked by visit_item
            hir::ItemMod(..) => {}
            // Reexports are handled in visit_mod
            hir::ItemUse(..) => {}
            // Visit everything
            hir::ItemConst(..) | hir::ItemStatic(..) | hir::ItemFn(..) |
            hir::ItemTrait(..) | hir::ItemTy(..) | hir::ItemImpl(_, _, _, Some(..), _, _) => {
                if item_level.is_some() {
                    self.reach().visit_item(item);
                }
            }
            // Visit everything, but enum variants have their own levels
            hir::ItemEnum(ref def, ref generics) => {
                if item_level.is_some() {
                    self.reach().visit_generics(generics);
                }
                for variant in &def.variants {
                    if self.get(variant.node.data.id()).is_some() {
                        for field in variant.node.data.fields() {
                            self.reach().visit_struct_field(field);
                        }
                        // Corner case: if the variant is reachable, but its
                        // enum is not, make the enum reachable as well.
                        self.update(item.id, Some(AccessLevel::Reachable));
                    }
                }
            }
            // Visit everything, but foreign items have their own levels
            hir::ItemForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if self.get(foreign_item.id).is_some() {
                        self.reach().visit_foreign_item(foreign_item);
                    }
                }
            }
            // Visit everything except for private fields
            hir::ItemStruct(ref struct_def, ref generics) => {
                if item_level.is_some() {
                    self.reach().visit_generics(generics);
                    for field in struct_def.fields() {
                        if self.get(field.id).is_some() {
                            self.reach().visit_struct_field(field);
                        }
                    }
                }
            }
            // The interface is empty
            hir::ItemDefaultImpl(..) => {}
            // Visit everything except for private impl items
            hir::ItemImpl(_, _, ref generics, None, _, ref impl_items) => {
                if item_level.is_some() {
                    self.reach().visit_generics(generics);
                    for impl_item in impl_items {
                        if self.get(impl_item.id).is_some() {
                            self.reach().visit_impl_item(impl_item);
                        }
                    }
                }
            }
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
            if let Some(exports) = self.export_map.get(&id) {
                for export in exports {
                    if let Some(node_id) = self.tcx.map.as_local_node_id(export.def_id) {
                        self.update(node_id, Some(AccessLevel::Exported));
                    }
                }
            }
        }

        intravisit::walk_mod(self, m);
    }

    fn visit_macro_def(&mut self, md: &'v hir::MacroDef) {
        self.update(md.id, Some(AccessLevel::Public));
    }
}

impl<'b, 'a, 'tcx: 'a> ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
    // Make the type hidden under a type alias reachable
    fn reach_aliased_type(&mut self, item: &hir::Item, path: &hir::Path) {
        if let hir::ItemTy(ref ty, ref generics) = item.node {
            // See `fn is_public_type_alias` for details
            self.visit_ty(ty);
            let provided_params = path.segments.last().unwrap().parameters.types().len();
            for ty_param in &generics.ty_params[provided_params..] {
                if let Some(ref default_ty) = ty_param.default {
                    self.visit_ty(default_ty);
                }
            }
        }
    }
}

impl<'b, 'a, 'tcx: 'a, 'v> Visitor<'v> for ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
    fn visit_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyPath(_, ref path) = ty.node {
            let def = self.ev.tcx.expect_def(ty.id);
            match def {
                Def::Struct(def_id) | Def::Enum(def_id) | Def::TyAlias(def_id) |
                Def::Trait(def_id) | Def::AssociatedTy(def_id, _) => {
                    if let Some(node_id) = self.ev.tcx.map.as_local_node_id(def_id) {
                        let item = self.ev.tcx.map.expect_item(node_id);
                        if let Def::TyAlias(..) = def {
                            // Type aliases are substituted. Associated type aliases are not
                            // substituted yet, but ideally they should be.
                            if self.ev.get(item.id).is_none() {
                                self.reach_aliased_type(item, path);
                            }
                        } else {
                            self.ev.update(item.id, Some(AccessLevel::Reachable));
                        }
                    }
                }

                _ => {}
            }
        }

        intravisit::walk_ty(self, ty);
    }

    fn visit_trait_ref(&mut self, trait_ref: &hir::TraitRef) {
        let def_id = self.ev.tcx.expect_def(trait_ref.ref_id).def_id();
        if let Some(node_id) = self.ev.tcx.map.as_local_node_id(def_id) {
            let item = self.ev.tcx.map.expect_item(node_id);
            self.ev.update(item.id, Some(AccessLevel::Reachable));
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

////////////////////////////////////////////////////////////////////////////////
/// The privacy visitor, where privacy checks take place (violations reported)
////////////////////////////////////////////////////////////////////////////////

struct PrivacyVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    curitem: ast::NodeId,
    in_foreign: bool,
}

impl<'a, 'tcx> PrivacyVisitor<'a, 'tcx> {
    fn item_is_accessible(&self, did: DefId) -> bool {
        match self.tcx.map.as_local_node_id(did) {
            Some(node_id) =>
                ty::Visibility::from_hir(&self.tcx.map.expect_item(node_id).vis, node_id, self.tcx),
            None => self.tcx.sess.cstore.visibility(did),
        }.is_accessible_from(self.curitem, &self.tcx.map)
    }

    // Checks that a field is in scope.
    fn check_field(&mut self, span: Span, def: ty::AdtDef<'tcx>, field: ty::FieldDef<'tcx>) {
        if def.adt_kind() == ty::AdtKind::Struct &&
           !field.vis.is_accessible_from(self.curitem, &self.tcx.map) {
            span_err!(self.tcx.sess, span, E0451, "field `{}` of struct `{}` is private",
                      field.name, self.tcx.item_path_str(def.did));
        }
    }

    // Checks that a method is in scope.
    fn check_method(&mut self, span: Span, method_def_id: DefId) {
        match self.tcx.impl_or_trait_item(method_def_id).container() {
            // Trait methods are always all public. The only controlling factor
            // is whether the trait itself is accessible or not.
            ty::TraitContainer(trait_def_id) if !self.item_is_accessible(trait_def_id) => {
                let msg = format!("source trait `{}` is private",
                                  self.tcx.item_path_str(trait_def_id));
                self.tcx.sess.span_err(span, &msg);
            }
            _ => {}
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for PrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        let tcx = self.tcx;
        self.visit_item(tcx.map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &hir::Item) {
        let orig_curitem = replace(&mut self.curitem, item.id);
        intravisit::walk_item(self, item);
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprMethodCall(..) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let method = self.tcx.tables.borrow().method_map[&method_call];
                self.check_method(expr.span, method.def_id);
            }
            hir::ExprStruct(..) => {
                let adt = self.tcx.expr_ty(expr).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(self.tcx.expect_def(expr.id));
                // RFC 736: ensure all unmentioned fields are visible.
                // Rather than computing the set of unmentioned fields
                // (i.e. `all_fields - fields`), just check them all.
                for field in &variant.fields {
                    self.check_field(expr.span, adt, field);
                }
            }
            hir::ExprPath(..) => {

                if let Def::Struct(..) = self.tcx.expect_def(expr.id) {
                    let expr_ty = self.tcx.expr_ty(expr);
                    let def = match expr_ty.sty {
                        ty::TyFnDef(_, _, &ty::BareFnTy { sig: ty::Binder(ty::FnSig {
                            output: ty::FnConverging(ty), ..
                        }), ..}) => ty,
                        _ => expr_ty
                    }.ty_adt_def().unwrap();
                    let any_priv = def.struct_variant().fields.iter().any(|f| {
                        !f.vis.is_accessible_from(self.curitem, &self.tcx.map)
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
            PatKind::Struct(_, ref fields, _) => {
                let adt = self.tcx.pat_ty(pattern).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(self.tcx.expect_def(pattern.id));
                for field in fields {
                    self.check_field(pattern.span, adt, variant.field_named(field.node.name));
                }
            }
            PatKind::TupleStruct(_, ref fields, ddpos) => {
                match self.tcx.pat_ty(pattern).sty {
                    ty::TyStruct(def, _) => {
                        let expected_len = def.struct_variant().fields.len();
                        for (i, field) in fields.iter().enumerate_and_adjust(expected_len, ddpos) {
                            if let PatKind::Wild = field.node {
                                continue
                            }
                            self.check_field(field.span, def, &def.struct_variant().fields[i]);
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
}

///////////////////////////////////////////////////////////////////////////////
/// Obsolete visitors for checking for private items in public interfaces.
/// These visitors are supposed to be kept in frozen state and produce an
/// "old error node set". For backward compatibility the new visitor reports
/// warnings instead of hard errors when the erroneous node is not in this old set.
///////////////////////////////////////////////////////////////////////////////

struct ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
        let did = match self.tcx.expect_def(path_id) {
            Def::PrimTy(..) | Def::SelfTy(..) => return false,
            def => def.def_id(),
        };

        // A path can only be private if:
        // it's in this crate...
        if let Some(node_id) = self.tcx.map.as_local_node_id(did) {
            // .. and it corresponds to a private type in the AST (this returns
            // None for type parameters)
            match self.tcx.map.find(node_id) {
                Some(hir::map::NodeItem(ref item)) => item.vis != hir::Public,
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

    fn item_is_public(&self, id: &ast::NodeId, vis: &hir::Visibility) -> bool {
        self.access_levels.is_reachable(*id) || *vis == hir::Public
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
        let tcx = self.tcx;
        self.visit_item(tcx.map.expect_item(item.id))
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
                    visitor.visit_ty(&self_);
                    self_contains_private = visitor.contains_private;
                    self_is_public_path = visitor.outer_type_is_public_path;
                }

                // miscellaneous info about the impl

                // `true` iff this is `impl Private for ...`.
                let not_private_trait =
                    trait_ref.as_ref().map_or(true, // no trait counts as public trait
                                              |tr| {
                        let did = self.tcx.expect_def(tr.ref_id).def_id();

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
                                        if self.item_is_public(&impl_item.id, &impl_item.vis) =>
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
                                if self.item_is_public(&impl_item.id, &impl_item.vis) {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                            }
                            hir::ImplItemKind::Method(ref sig, _) => {
                                if !sig.decl.has_self() &&
                                        self.item_is_public(&impl_item.id, &impl_item.vis) {
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
            _ if !self.item_is_public(&item.id, &item.vis) => {
                return;
            }

            _ => {}
        }

        // We've carefully constructed it so that if we're here, then
        // any `visit_ty`'s will be called on things that are in
        // public signatures, i.e. things that we're interested in for
        // this visitor.
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
                    self.visit_ty(&eq_pred.ty);
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
        if s.vis == hir::Public || self.in_variant {
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
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    /// The visitor checks that each component type is at least this visible
    required_visibility: ty::Visibility,
    /// The visibility of the least visible component that has been visited
    min_visibility: ty::Visibility,
    old_error_set: &'a NodeSet,
}

impl<'a, 'tcx: 'a> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, old_error_set: &'a NodeSet) -> Self {
        SearchInterfaceForPrivateItemsVisitor {
            tcx: tcx,
            min_visibility: ty::Visibility::Public,
            required_visibility: ty::Visibility::PrivateExternal,
            old_error_set: old_error_set,
        }
    }
}

impl<'a, 'tcx: 'a> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    // Return the visibility of the type alias's least visible component type when substituted
    fn substituted_alias_visibility(&self, item: &hir::Item, path: &hir::Path)
                                    -> Option<ty::Visibility> {
        // We substitute type aliases only when determining impl publicity
        // FIXME: This will probably change and all type aliases will be substituted,
        // requires an amendment to RFC 136.
        if self.required_visibility != ty::Visibility::PrivateExternal {
            return None;
        }
        // Type alias is considered public if the aliased type is
        // public, even if the type alias itself is private. So, something
        // like `type A = u8; pub fn f() -> A {...}` doesn't cause an error.
        if let hir::ItemTy(ref ty, ref generics) = item.node {
            let mut check = SearchInterfaceForPrivateItemsVisitor {
                min_visibility: ty::Visibility::Public, ..*self
            };
            check.visit_ty(ty);
            // If a private type alias with default type parameters is used in public
            // interface we must ensure, that the defaults are public if they are actually used.
            // ```
            // type Alias<T = Private> = T;
            // pub fn f() -> Alias {...} // `Private` is implicitly used here, so it must be public
            // ```
            let provided_params = path.segments.last().unwrap().parameters.types().len();
            for ty_param in &generics.ty_params[provided_params..] {
                if let Some(ref default_ty) = ty_param.default {
                    check.visit_ty(default_ty);
                }
            }
            Some(check.min_visibility)
        } else {
            None
        }
    }
}

impl<'a, 'tcx: 'a, 'v> Visitor<'v> for SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyPath(_, ref path) = ty.node {
            match self.tcx.expect_def(ty.id) {
                Def::PrimTy(..) | Def::SelfTy(..) | Def::TyParam(..) => {
                    // Public
                }
                Def::AssociatedTy(..)
                    if self.required_visibility == ty::Visibility::PrivateExternal => {
                    // Conservatively approximate the whole type alias as public without
                    // recursing into its components when determining impl publicity.
                    // For example, `impl <Type as Trait>::Alias {...}` may be a public impl
                    // even if both `Type` and `Trait` are private.
                    // Ideally, associated types should be substituted in the same way as
                    // free type aliases, but this isn't done yet.
                    return
                }
                Def::Struct(def_id) | Def::Enum(def_id) | Def::TyAlias(def_id) |
                Def::Trait(def_id) | Def::AssociatedTy(def_id, _) => {
                    // Non-local means public (private items can't leave their crate, modulo bugs)
                    if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
                        let item = self.tcx.map.expect_item(node_id);
                        let vis = match self.substituted_alias_visibility(item, path) {
                            Some(vis) => vis,
                            None => ty::Visibility::from_hir(&item.vis, node_id, self.tcx),
                        };

                        if !vis.is_at_least(self.min_visibility, &self.tcx.map) {
                            self.min_visibility = vis;
                        }
                        if !vis.is_at_least(self.required_visibility, &self.tcx.map) {
                            if self.tcx.sess.features.borrow().pub_restricted ||
                               self.old_error_set.contains(&ty.id) {
                                span_err!(self.tcx.sess, ty.span, E0446,
                                          "private type in public interface");
                            } else {
                                self.tcx.sess.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                                       node_id,
                                                       ty.span,
                                                       format!("private type in public interface"));
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
        // Non-local means public (private items can't leave their crate, modulo bugs)
        let def_id = self.tcx.expect_def(trait_ref.ref_id).def_id();
        if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
            let item = self.tcx.map.expect_item(node_id);
            let vis = ty::Visibility::from_hir(&item.vis, node_id, self.tcx);

            if !vis.is_at_least(self.min_visibility, &self.tcx.map) {
                self.min_visibility = vis;
            }
            if !vis.is_at_least(self.required_visibility, &self.tcx.map) {
                if self.tcx.sess.features.borrow().pub_restricted ||
                   self.old_error_set.contains(&trait_ref.ref_id) {
                    span_err!(self.tcx.sess, trait_ref.path.span, E0445,
                              "private trait in public interface");
                } else {
                    self.tcx.sess.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                           node_id,
                                           trait_ref.path.span,
                                           "private trait in public interface (error E0445)"
                                                .to_string());
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
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    old_error_set: &'a NodeSet,
}

impl<'a, 'tcx> PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    // A type is considered public if it doesn't contain any private components
    fn ty_visibility(&self, ty: &hir::Ty) -> ty::Visibility {
        let mut check = SearchInterfaceForPrivateItemsVisitor::new(self.tcx, self.old_error_set);
        check.visit_ty(ty);
        check.min_visibility
    }

    // A trait reference is considered public if it doesn't contain any private components
    fn trait_ref_visibility(&self, trait_ref: &hir::TraitRef) -> ty::Visibility {
        let mut check = SearchInterfaceForPrivateItemsVisitor::new(self.tcx, self.old_error_set);
        check.visit_trait_ref(trait_ref);
        check.min_visibility
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        let min = |vis1: ty::Visibility, vis2| {
            if vis1.is_at_least(vis2, &self.tcx.map) { vis2 } else { vis1 }
        };

        let mut check = SearchInterfaceForPrivateItemsVisitor::new(self.tcx, self.old_error_set);
        let item_visibility = ty::Visibility::from_hir(&item.vis, item.id, self.tcx);

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
                check.required_visibility = item_visibility;
                check.visit_item(item);
            }
            // Subitems of foreign modules have their own publicity
            hir::ItemForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    check.required_visibility =
                        ty::Visibility::from_hir(&foreign_item.vis, item.id, self.tcx);
                    check.visit_foreign_item(foreign_item);
                }
            }
            // Subitems of structs have their own publicity
            hir::ItemStruct(ref struct_def, ref generics) => {
                check.required_visibility = item_visibility;
                check.visit_generics(generics);

                for field in struct_def.fields() {
                    let field_visibility = ty::Visibility::from_hir(&field.vis, item.id, self.tcx);
                    check.required_visibility = min(item_visibility, field_visibility);
                    check.visit_struct_field(field);
                }
            }
            // The interface is empty
            hir::ItemDefaultImpl(..) => {}
            // An inherent impl is public when its type is public
            // Subitems of inherent impls have their own publicity
            hir::ItemImpl(_, _, ref generics, None, ref ty, ref impl_items) => {
                let ty_vis = self.ty_visibility(ty);
                check.required_visibility = ty_vis;
                check.visit_generics(generics);

                for impl_item in impl_items {
                    let impl_item_vis =
                        ty::Visibility::from_hir(&impl_item.vis, item.id, self.tcx);
                    check.required_visibility = min(impl_item_vis, ty_vis);
                    check.visit_impl_item(impl_item);
                }
            }
            // A trait impl is public when both its type and its trait are public
            // Subitems of trait impls have inherited publicity
            hir::ItemImpl(_, _, ref generics, Some(ref trait_ref), ref ty, ref impl_items) => {
                let vis = min(self.ty_visibility(ty), self.trait_ref_visibility(trait_ref));
                check.required_visibility = vis;
                check.visit_generics(generics);
                for impl_item in impl_items {
                    check.visit_impl_item(impl_item);
                }
            }
        }
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             export_map: &def::ExportMap)
                             -> AccessLevels {
    let _task = tcx.dep_graph.in_task(DepNode::Privacy);

    let krate = tcx.map.krate();

    // Use the parent map to check the privacy of everything
    let mut visitor = PrivacyVisitor {
        curitem: ast::DUMMY_NODE_ID,
        in_foreign: false,
        tcx: tcx,
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
