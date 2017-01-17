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
#![deny(warnings)]

#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

extern crate rustc;
#[macro_use] extern crate syntax;
extern crate syntax_pos;

use rustc::dep_graph::DepNode;
use rustc::hir::{self, PatKind};
use rustc::hir::def::{self, Def, CtorKind};
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::itemlikevisit::DeepVisitor;
use rustc::hir::pat_util::EnumerateAndAdjustIterator;
use rustc::lint;
use rustc::middle::privacy::{AccessLevel, AccessLevels};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable};
use rustc::ty::fold::TypeVisitor;
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
    item_def_id: DefId,
    ev: &'b mut EmbargoVisitor<'a, 'tcx>,
}

impl<'a, 'tcx> EmbargoVisitor<'a, 'tcx> {
    fn item_ty_level(&self, item_def_id: DefId) -> Option<AccessLevel> {
        let ty_def_id = match self.tcx.item_type(item_def_id).sty {
            ty::TyAdt(adt, _) => adt.did,
            ty::TyDynamic(ref obj, ..) if obj.principal().is_some() =>
                obj.principal().unwrap().def_id(),
            ty::TyProjection(ref proj) => proj.trait_ref.def_id,
            _ => return Some(AccessLevel::Public)
        };
        if let Some(node_id) = self.tcx.map.as_local_node_id(ty_def_id) {
            self.get(node_id)
        } else {
            Some(AccessLevel::Public)
        }
    }

    fn impl_trait_level(&self, impl_def_id: DefId) -> Option<AccessLevel> {
        if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_def_id) {
            if let Some(node_id) = self.tcx.map.as_local_node_id(trait_ref.def_id) {
                return self.get(node_id);
            }
        }
        Some(AccessLevel::Public)
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

    fn reach<'b>(&'b mut self, item_id: ast::NodeId)
                 -> ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
        ReachEverythingInTheInterfaceVisitor {
            item_def_id: self.tcx.map.local_def_id(item_id),
            ev: self,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for EmbargoVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.map)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let inherited_item_level = match item.node {
            // Impls inherit level from their types and traits
            hir::ItemImpl(..) => {
                let def_id = self.tcx.map.local_def_id(item.id);
                cmp::min(self.item_ty_level(def_id), self.impl_trait_level(def_id))
            }
            hir::ItemDefaultImpl(..) => {
                let def_id = self.tcx.map.local_def_id(item.id);
                self.impl_trait_level(def_id)
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
            hir::ItemImpl(.., None, _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    if impl_item_ref.vis == hir::Public {
                        self.update(impl_item_ref.id.node_id, item_level);
                    }
                }
            }
            hir::ItemImpl(.., Some(_), _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    self.update(impl_item_ref.id.node_id, item_level);
                }
            }
            hir::ItemTrait(.., ref trait_item_refs) => {
                for trait_item_ref in trait_item_refs {
                    self.update(trait_item_ref.id.node_id, item_level);
                }
            }
            hir::ItemStruct(ref def, _) | hir::ItemUnion(ref def, _) => {
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
            // The interface is empty
            hir::ItemDefaultImpl(..) => {}
            // Visit everything
            hir::ItemConst(..) | hir::ItemStatic(..) |
            hir::ItemFn(..) | hir::ItemTy(..) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates().item_type();
                }
            }
            hir::ItemTrait(.., ref trait_item_refs) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();

                    for trait_item_ref in trait_item_refs {
                        let mut reach = self.reach(trait_item_ref.id.node_id);
                        reach.generics().predicates();

                        if trait_item_ref.kind == hir::AssociatedItemKind::Type &&
                           !trait_item_ref.defaultness.has_value() {
                            // No type to visit.
                        } else {
                            reach.item_type();
                        }
                    }
                }
            }
            // Visit everything except for private impl items
            hir::ItemImpl(.., ref trait_ref, _, ref impl_item_refs) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates().impl_trait_ref();

                    for impl_item_ref in impl_item_refs {
                        let id = impl_item_ref.id.node_id;
                        if trait_ref.is_some() || self.get(id).is_some() {
                            self.reach(id).generics().predicates().item_type();
                        }
                    }
                }
            }

            // Visit everything, but enum variants have their own levels
            hir::ItemEnum(ref def, _) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();
                }
                for variant in &def.variants {
                    if self.get(variant.node.data.id()).is_some() {
                        for field in variant.node.data.fields() {
                            self.reach(field.id).item_type();
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
                        self.reach(foreign_item.id).generics().predicates().item_type();
                    }
                }
            }
            // Visit everything except for private fields
            hir::ItemStruct(ref struct_def, _) |
            hir::ItemUnion(ref struct_def, _) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();
                    for field in struct_def.fields() {
                        if self.get(field.id).is_some() {
                            self.reach(field.id).item_type();
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

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        let orig_level = replace(&mut self.prev_level, None);

        // Blocks can have public items, for example impls, but they always
        // start as completely private regardless of publicity of a function,
        // constant, type, field, etc. in which this block resides
        intravisit::walk_block(self, b);

        self.prev_level = orig_level;
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod, _sp: Span, id: ast::NodeId) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_level.is_some() {
            if let Some(exports) = self.export_map.get(&id) {
                for export in exports {
                    if let Some(node_id) = self.tcx.map.as_local_node_id(export.def.def_id()) {
                        self.update(node_id, Some(AccessLevel::Exported));
                    }
                }
            }
        }

        intravisit::walk_mod(self, m, id);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
        self.update(md.id, Some(AccessLevel::Public));
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        if let hir::TyImplTrait(..) = ty.node {
            if self.get(ty.id).is_some() {
                // Reach the (potentially private) type and the API being exposed.
                self.reach(ty.id).item_type().predicates();
            }
        }

        intravisit::walk_ty(self, ty);
    }
}

impl<'b, 'a, 'tcx> ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
    fn generics(&mut self) -> &mut Self {
        self.ev.tcx.item_generics(self.item_def_id).visit_with(self);
        self
    }

    fn predicates(&mut self) -> &mut Self {
        self.ev.tcx.item_predicates(self.item_def_id).visit_with(self);
        self
    }

    fn item_type(&mut self) -> &mut Self {
        self.ev.tcx.item_type(self.item_def_id).visit_with(self);
        self
    }

    fn impl_trait_ref(&mut self) -> &mut Self {
        self.ev.tcx.impl_trait_ref(self.item_def_id).visit_with(self);
        self
    }
}

impl<'b, 'a, 'tcx> TypeVisitor<'tcx> for ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        let ty_def_id = match ty.sty {
            ty::TyAdt(adt, _) => Some(adt.did),
            ty::TyDynamic(ref obj, ..) => obj.principal().map(|p| p.def_id()),
            ty::TyProjection(ref proj) => Some(proj.trait_ref.def_id),
            ty::TyFnDef(def_id, ..) |
            ty::TyAnon(def_id, _) => Some(def_id),
            _ => None
        };

        if let Some(def_id) = ty_def_id {
            if let Some(node_id) = self.ev.tcx.map.as_local_node_id(def_id) {
                self.ev.update(node_id, Some(AccessLevel::Reachable));
            }
        }

        ty.super_visit_with(self)
    }

    fn visit_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>) -> bool {
        if let Some(node_id) = self.ev.tcx.map.as_local_node_id(trait_ref.def_id) {
            let item = self.ev.tcx.map.expect_item(node_id);
            self.ev.update(item.id, Some(AccessLevel::Reachable));
        }

        trait_ref.super_visit_with(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy visitor, where privacy checks take place (violations reported)
////////////////////////////////////////////////////////////////////////////////

struct PrivacyVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    curitem: DefId,
    in_foreign: bool,
    tables: &'a ty::Tables<'tcx>,
}

impl<'a, 'tcx> PrivacyVisitor<'a, 'tcx> {
    fn item_is_accessible(&self, did: DefId) -> bool {
        match self.tcx.map.as_local_node_id(did) {
            Some(node_id) =>
                ty::Visibility::from_hir(&self.tcx.map.expect_item(node_id).vis, node_id, self.tcx),
            None => self.tcx.sess.cstore.visibility(did),
        }.is_accessible_from(self.curitem, self.tcx)
    }

    // Checks that a field is in scope.
    fn check_field(&mut self, span: Span, def: &'tcx ty::AdtDef, field: &'tcx ty::FieldDef) {
        if !def.is_enum() && !field.vis.is_accessible_from(self.curitem, self.tcx) {
            struct_span_err!(self.tcx.sess, span, E0451, "field `{}` of {} `{}` is private",
                      field.name, def.variant_descr(), self.tcx.item_path_str(def.did))
                .span_label(span, &format!("field `{}` is private", field.name))
                .emit();
        }
    }

    // Checks that a method is in scope.
    fn check_method(&mut self, span: Span, method_def_id: DefId) {
        match self.tcx.associated_item(method_def_id).container {
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

impl<'a, 'tcx> Visitor<'tcx> for PrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.map)
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.map.body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let orig_curitem = replace(&mut self.curitem, self.tcx.map.local_def_id(item.id));
        intravisit::walk_item(self, item);
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprMethodCall(..) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let method = self.tables.method_map[&method_call];
                self.check_method(expr.span, method.def_id);
            }
            hir::ExprStruct(ref qpath, ref expr_fields, _) => {
                let def = self.tables.qpath_def(qpath, expr.id);
                let adt = self.tables.expr_ty(expr).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(def);
                // RFC 736: ensure all unmentioned fields are visible.
                // Rather than computing the set of unmentioned fields
                // (i.e. `all_fields - fields`), just check them all,
                // unless the ADT is a union, then unmentioned fields
                // are not checked.
                if adt.is_union() {
                    for expr_field in expr_fields {
                        self.check_field(expr.span, adt, variant.field_named(expr_field.name.node));
                    }
                } else {
                    for field in &variant.fields {
                        let expr_field = expr_fields.iter().find(|f| f.name.node == field.name);
                        let span = if let Some(f) = expr_field { f.span } else { expr.span };
                        self.check_field(span, adt, field);
                    }
                }
            }
            hir::ExprPath(hir::QPath::Resolved(_, ref path)) => {
                if let Def::StructCtor(_, CtorKind::Fn) = path.def {
                    let adt_def = self.tcx.expect_variant_def(path.def);
                    let private_indexes = adt_def.fields.iter().enumerate().filter(|&(_, field)| {
                        !field.vis.is_accessible_from(self.curitem, self.tcx)
                    }).map(|(i, _)| i).collect::<Vec<_>>();

                    if !private_indexes.is_empty() {
                        let mut error = struct_span_err!(self.tcx.sess, expr.span, E0450,
                                                         "cannot invoke tuple struct constructor \
                                                          with private fields");
                        error.span_label(expr.span,
                                         &format!("cannot construct with a private field"));

                        if let Some(node_id) = self.tcx.map.as_local_node_id(adt_def.did) {
                            let node = self.tcx.map.find(node_id);
                            if let Some(hir::map::NodeStructCtor(vdata)) = node {
                                for i in private_indexes {
                                    error.span_label(vdata.fields()[i].span,
                                                     &format!("private field declared here"));
                                }
                            }
                        }
                        error.emit();
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pattern: &'tcx hir::Pat) {
        // Foreign functions do not have their patterns mapped in the def_map,
        // and there's nothing really relevant there anyway, so don't bother
        // checking privacy. If you can name the type then you can pass it to an
        // external C function anyway.
        if self.in_foreign { return }

        match pattern.node {
            PatKind::Struct(ref qpath, ref fields, _) => {
                let def = self.tables.qpath_def(qpath, pattern.id);
                let adt = self.tables.pat_ty(pattern).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(def);
                for field in fields {
                    self.check_field(field.span, adt, variant.field_named(field.node.name));
                }
            }
            PatKind::TupleStruct(_, ref fields, ddpos) => {
                match self.tables.pat_ty(pattern).sty {
                    // enum fields have no privacy at this time
                    ty::TyAdt(def, _) if !def.is_enum() => {
                        let expected_len = def.struct_variant().fields.len();
                        for (i, field) in fields.iter().enumerate_and_adjust(expected_len, ddpos) {
                            if let PatKind::Wild = field.node {
                                continue
                            }
                            self.check_field(field.span, def, &def.struct_variant().fields[i]);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        intravisit::walk_pat(self, pattern);
    }

    fn visit_foreign_item(&mut self, fi: &'tcx hir::ForeignItem) {
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
    fn path_is_private_type(&self, path: &hir::Path) -> bool {
        let did = match path.def {
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
            if self.path_is_private_type(&trait_ref.trait_ref.path) {
                self.old_error_set.insert(trait_ref.trait_ref.ref_id);
            }
        }
    }

    fn item_is_public(&self, id: &ast::NodeId, vis: &hir::Visibility) -> bool {
        self.access_levels.is_reachable(*id) || *vis == hir::Public
    }
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

    fn visit_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyPath(hir::QPath::Resolved(_, ref path)) = ty.node {
            if self.inner.path_is_private_type(path) {
                self.contains_private = true;
                // found what we're looking for so let's stop
                // working.
                return
            }
        }
        if let hir::TyPath(_) = ty.node {
            if self.at_outer_type {
                self.outer_type_is_public_path = true;
            }
        }
        self.at_outer_type = false;
        intravisit::walk_ty(self, ty)
    }

    // don't want to recurse into [, .. expr]
    fn visit_expr(&mut self, _: &hir::Expr) {}
}

impl<'a, 'tcx> Visitor<'tcx> for ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.map)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            // contents of a private mod can be reexported, so we need
            // to check internals.
            hir::ItemMod(_) => {}

            // An `extern {}` doesn't introduce a new privacy
            // namespace (the contents have their own privacies).
            hir::ItemForeignMod(_) => {}

            hir::ItemTrait(.., ref bounds, _) => {
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
            hir::ItemImpl(.., ref g, ref trait_ref, ref self_, ref impl_item_refs) => {
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
                        let did = tr.path.def.def_id();

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
                    impl_item_refs.iter()
                                 .any(|impl_item_ref| {
                                     let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
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
                            for impl_item_ref in impl_item_refs {
                                // This is where we choose whether to walk down
                                // further into the impl to check its items. We
                                // should only walk into public items so that we
                                // don't erroneously report errors for private
                                // types in private items.
                                let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
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
                            for impl_item_ref in impl_item_refs {
                                let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
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
                    for impl_item_ref in impl_item_refs {
                        if self.item_is_public(&impl_item_ref.id.node_id, &impl_item_ref.vis) {
                            let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
                            match impl_item_ref.kind {
                                hir::AssociatedItemKind::Const => {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                                hir::AssociatedItemKind::Method { has_self: false } => {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                                _ => {}
                            }
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

    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
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
                    self.visit_ty(&eq_pred.rhs_ty);
                }
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem) {
        if self.access_levels.is_reachable(item.id) {
            intravisit::walk_foreign_item(self, item)
        }
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        if let hir::TyPath(hir::QPath::Resolved(_, ref path)) = t.node {
            if self.path_is_private_type(path) {
                self.old_error_set.insert(t.id);
            }
        }
        intravisit::walk_ty(self, t)
    }

    fn visit_variant(&mut self,
                     v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     item_id: ast::NodeId) {
        if self.access_levels.is_reachable(v.node.data.id()) {
            self.in_variant = true;
            intravisit::walk_variant(self, v, g, item_id);
            self.in_variant = false;
        }
    }

    fn visit_struct_field(&mut self, s: &'tcx hir::StructField) {
        if s.vis == hir::Public || self.in_variant {
            intravisit::walk_struct_field(self, s);
        }
    }

    // we don't need to introspect into these at all: an
    // expression/block context can't possibly contain exported things.
    // (Making them no-ops stops us from traversing the whole AST without
    // having to be super careful about our `walk_...` calls above.)
    fn visit_block(&mut self, _: &'tcx hir::Block) {}
    fn visit_expr(&mut self, _: &'tcx hir::Expr) {}
}

///////////////////////////////////////////////////////////////////////////////
/// SearchInterfaceForPrivateItemsVisitor traverses an item's interface and
/// finds any private components in it.
/// PrivateItemsInPublicInterfacesVisitor ensures there are no private types
/// and traits in public interfaces.
///////////////////////////////////////////////////////////////////////////////

struct SearchInterfaceForPrivateItemsVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    item_def_id: DefId,
    span: Span,
    /// The visitor checks that each component type is at least this visible
    required_visibility: ty::Visibility,
    /// The visibility of the least visible component that has been visited
    min_visibility: ty::Visibility,
    has_old_errors: bool,
}

impl<'a, 'tcx: 'a> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn generics(&mut self) -> &mut Self {
        self.tcx.item_generics(self.item_def_id).visit_with(self);
        self
    }

    fn predicates(&mut self) -> &mut Self {
        self.tcx.item_predicates(self.item_def_id).visit_with(self);
        self
    }

    fn item_type(&mut self) -> &mut Self {
        self.tcx.item_type(self.item_def_id).visit_with(self);
        self
    }

    fn impl_trait_ref(&mut self) -> &mut Self {
        self.tcx.impl_trait_ref(self.item_def_id).visit_with(self);
        self
    }
}

impl<'a, 'tcx: 'a> TypeVisitor<'tcx> for SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        let ty_def_id = match ty.sty {
            ty::TyAdt(adt, _) => Some(adt.did),
            ty::TyDynamic(ref obj, ..) => obj.principal().map(|p| p.def_id()),
            ty::TyProjection(ref proj) => {
                if self.required_visibility == ty::Visibility::Invisible {
                    // Conservatively approximate the whole type alias as public without
                    // recursing into its components when determining impl publicity.
                    // For example, `impl <Type as Trait>::Alias {...}` may be a public impl
                    // even if both `Type` and `Trait` are private.
                    // Ideally, associated types should be substituted in the same way as
                    // free type aliases, but this isn't done yet.
                    return false;
                }

                Some(proj.trait_ref.def_id)
            }
            _ => None
        };

        if let Some(def_id) = ty_def_id {
            // Non-local means public (private items can't leave their crate, modulo bugs)
            if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
                let item = self.tcx.map.expect_item(node_id);
                let vis = ty::Visibility::from_hir(&item.vis, node_id, self.tcx);

                if !vis.is_at_least(self.min_visibility, self.tcx) {
                    self.min_visibility = vis;
                }
                if !vis.is_at_least(self.required_visibility, self.tcx) {
                    if self.tcx.sess.features.borrow().pub_restricted || self.has_old_errors {
                        let mut err = struct_span_err!(self.tcx.sess, self.span, E0446,
                            "private type `{}` in public interface", ty);
                        err.span_label(self.span, &format!("can't leak private type"));
                        err.emit();
                    } else {
                        self.tcx.sess.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                               node_id,
                                               self.span,
                                               format!("private type `{}` in public \
                                                        interface (error E0446)", ty));
                    }
                }
            }
        }

        if let ty::TyProjection(ref proj) = ty.sty {
            // Avoid calling `visit_trait_ref` below on the trait,
            // as we have already checked the trait itself above.
            proj.trait_ref.super_visit_with(self)
        } else {
            ty.super_visit_with(self)
        }
    }

    fn visit_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>) -> bool {
        // Non-local means public (private items can't leave their crate, modulo bugs)
        if let Some(node_id) = self.tcx.map.as_local_node_id(trait_ref.def_id) {
            let item = self.tcx.map.expect_item(node_id);
            let vis = ty::Visibility::from_hir(&item.vis, node_id, self.tcx);

            if !vis.is_at_least(self.min_visibility, self.tcx) {
                self.min_visibility = vis;
            }
            if !vis.is_at_least(self.required_visibility, self.tcx) {
                if self.tcx.sess.features.borrow().pub_restricted || self.has_old_errors {
                    struct_span_err!(self.tcx.sess, self.span, E0445,
                                     "private trait `{}` in public interface", trait_ref)
                        .span_label(self.span, &format!(
                                    "private trait can't be public"))
                        .emit();
                } else {
                    self.tcx.sess.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                           node_id,
                                           self.span,
                                           format!("private trait `{}` in public \
                                                    interface (error E0445)", trait_ref));
                }
            }
        }

        trait_ref.super_visit_with(self)
    }
}

struct PrivateItemsInPublicInterfacesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    old_error_set: &'a NodeSet,
    inner_visibility: ty::Visibility,
}

impl<'a, 'tcx> PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn check(&self, item_id: ast::NodeId, required_visibility: ty::Visibility)
             -> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
        let mut has_old_errors = false;

        // Slow path taken only if there any errors in the crate.
        for &id in self.old_error_set {
            // Walk up the nodes until we find `item_id` (or we hit a root).
            let mut id = id;
            loop {
                if id == item_id {
                    has_old_errors = true;
                    break;
                }
                let parent = self.tcx.map.get_parent_node(id);
                if parent == id {
                    break;
                }
                id = parent;
            }

            if has_old_errors {
                break;
            }
        }

        SearchInterfaceForPrivateItemsVisitor {
            tcx: self.tcx,
            item_def_id: self.tcx.map.local_def_id(item_id),
            span: self.tcx.map.span(item_id),
            min_visibility: ty::Visibility::Public,
            required_visibility: required_visibility,
            has_old_errors: has_old_errors,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.map)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let tcx = self.tcx;
        let min = |vis1: ty::Visibility, vis2| {
            if vis1.is_at_least(vis2, tcx) { vis2 } else { vis1 }
        };

        let item_visibility = ty::Visibility::from_hir(&item.vis, item.id, tcx);

        match item.node {
            // Crates are always public
            hir::ItemExternCrate(..) => {}
            // All nested items are checked by visit_item
            hir::ItemMod(..) => {}
            // Checked in resolve
            hir::ItemUse(..) => {}
            // Subitems of these items have inherited publicity
            hir::ItemConst(..) | hir::ItemStatic(..) | hir::ItemFn(..) |
            hir::ItemTy(..) => {
                self.check(item.id, item_visibility).generics().predicates().item_type();

                // Recurse for e.g. `impl Trait` (see `visit_ty`).
                self.inner_visibility = item_visibility;
                intravisit::walk_item(self, item);
            }
            hir::ItemTrait(.., ref trait_item_refs) => {
                self.check(item.id, item_visibility).generics().predicates();

                for trait_item_ref in trait_item_refs {
                    let mut check = self.check(trait_item_ref.id.node_id, item_visibility);
                    check.generics().predicates();

                    if trait_item_ref.kind == hir::AssociatedItemKind::Type &&
                       !trait_item_ref.defaultness.has_value() {
                        // No type to visit.
                    } else {
                        check.item_type();
                    }
                }
            }
            hir::ItemEnum(ref def, _) => {
                self.check(item.id, item_visibility).generics().predicates();

                for variant in &def.variants {
                    for field in variant.node.data.fields() {
                        self.check(field.id, item_visibility).item_type();
                    }
                }
            }
            // Subitems of foreign modules have their own publicity
            hir::ItemForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    let vis = ty::Visibility::from_hir(&foreign_item.vis, item.id, tcx);
                    self.check(foreign_item.id, vis).generics().predicates().item_type();
                }
            }
            // Subitems of structs and unions have their own publicity
            hir::ItemStruct(ref struct_def, _) |
            hir::ItemUnion(ref struct_def, _) => {
                self.check(item.id, item_visibility).generics().predicates();

                for field in struct_def.fields() {
                    let field_visibility = ty::Visibility::from_hir(&field.vis, item.id, tcx);
                    self.check(field.id, min(item_visibility, field_visibility)).item_type();
                }
            }
            // The interface is empty
            hir::ItemDefaultImpl(..) => {}
            // An inherent impl is public when its type is public
            // Subitems of inherent impls have their own publicity
            hir::ItemImpl(.., None, _, ref impl_item_refs) => {
                let ty_vis =
                    self.check(item.id, ty::Visibility::Invisible).item_type().min_visibility;
                self.check(item.id, ty_vis).generics().predicates();

                for impl_item_ref in impl_item_refs {
                    let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
                    let impl_item_vis =
                        ty::Visibility::from_hir(&impl_item.vis, item.id, tcx);
                    self.check(impl_item.id, min(impl_item_vis, ty_vis))
                        .generics().predicates().item_type();

                    // Recurse for e.g. `impl Trait` (see `visit_ty`).
                    self.inner_visibility = impl_item_vis;
                    intravisit::walk_impl_item(self, impl_item);
                }
            }
            // A trait impl is public when both its type and its trait are public
            // Subitems of trait impls have inherited publicity
            hir::ItemImpl(.., Some(_), _, ref impl_item_refs) => {
                let vis = self.check(item.id, ty::Visibility::Invisible)
                              .item_type().impl_trait_ref().min_visibility;
                self.check(item.id, vis).generics().predicates();
                for impl_item_ref in impl_item_refs {
                    let impl_item = self.tcx.map.impl_item(impl_item_ref.id);
                    self.check(impl_item.id, vis).generics().predicates().item_type();

                    // Recurse for e.g. `impl Trait` (see `visit_ty`).
                    self.inner_visibility = vis;
                    intravisit::walk_impl_item(self, impl_item);
                }
            }
        }
    }

    fn visit_impl_item(&mut self, _impl_item: &'tcx hir::ImplItem) {
        // handled in `visit_item` above
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        if let hir::TyImplTrait(..) = ty.node {
            // Check the traits being exposed, as they're separate,
            // e.g. `impl Iterator<Item=T>` has two predicates,
            // `X: Iterator` and `<X as Iterator>::Item == T`,
            // where `X` is the `impl Iterator<Item=T>` itself,
            // stored in `item_predicates`, not in the `Ty` itself.
            self.check(ty.id, self.inner_visibility).predicates();
        }

        intravisit::walk_ty(self, ty);
    }

    // Don't recurse into expressions in array sizes or const initializers
    fn visit_expr(&mut self, _: &'tcx hir::Expr) {}
    // Don't recurse into patterns in function arguments
    fn visit_pat(&mut self, _: &'tcx hir::Pat) {}
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             export_map: &def::ExportMap)
                             -> AccessLevels {
    let _task = tcx.dep_graph.in_task(DepNode::Privacy);

    let krate = tcx.map.krate();

    // Use the parent map to check the privacy of everything
    let mut visitor = PrivacyVisitor {
        curitem: DefId::local(CRATE_DEF_INDEX),
        in_foreign: false,
        tcx: tcx,
        tables: &ty::Tables::empty(),
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
            inner_visibility: ty::Visibility::Public,
        };
        krate.visit_all_item_likes(&mut DeepVisitor::new(&mut visitor));
    }

    visitor.access_levels
}

__build_diagnostic_array! { librustc_privacy, DIAGNOSTICS }
