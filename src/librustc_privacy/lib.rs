#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(nll)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

#[macro_use] extern crate rustc;
#[macro_use] extern crate syntax;
extern crate rustc_typeck;
extern crate syntax_pos;
extern crate rustc_data_structures;

use rustc::hir::{self, Node, PatKind};
use rustc::hir::def::Def;
use rustc::hir::def_id::{CRATE_DEF_INDEX, LOCAL_CRATE, CrateNum, DefId};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::itemlikevisit::DeepVisitor;
use rustc::lint;
use rustc::middle::privacy::{AccessLevel, AccessLevels};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable, GenericParamDefKind};
use rustc::ty::fold::TypeVisitor;
use rustc::ty::query::Providers;
use rustc::ty::subst::UnpackedKind;
use rustc::util::nodemap::NodeSet;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use syntax::ast::{self, CRATE_NODE_ID, Ident};
use syntax::symbol::keywords;
use syntax_pos::Span;

use std::cmp;
use std::mem::replace;

mod diagnostics;

////////////////////////////////////////////////////////////////////////////////
/// Visitor used to determine if pub(restricted) is used anywhere in the crate.
///
/// This is done so that `private_in_public` warnings can be turned into hard errors
/// in crates that have been updated to use pub(restricted).
////////////////////////////////////////////////////////////////////////////////
struct PubRestrictedVisitor<'a, 'tcx: 'a> {
    tcx:  TyCtxt<'a, 'tcx, 'tcx>,
    has_pub_restricted: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for PubRestrictedVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }
    fn visit_vis(&mut self, vis: &'tcx hir::Visibility) {
        self.has_pub_restricted = self.has_pub_restricted || vis.node.is_pub_restricted();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the ast
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,

    // Accessibility levels for reachable nodes.
    access_levels: AccessLevels,
    // Previous accessibility level; `None` means unreachable.
    prev_level: Option<AccessLevel>,
    // Has something changed in the level map?
    changed: bool,
}

struct ReachEverythingInTheInterfaceVisitor<'b, 'a: 'b, 'tcx: 'a> {
    access_level: Option<AccessLevel>,
    item_def_id: DefId,
    ev: &'b mut EmbargoVisitor<'a, 'tcx>,
}

impl<'a, 'tcx> EmbargoVisitor<'a, 'tcx> {
    fn item_ty_level(&self, item_def_id: DefId) -> Option<AccessLevel> {
        let ty_def_id = match self.tcx.type_of(item_def_id).sty {
            ty::Adt(adt, _) => adt.did,
            ty::Foreign(did) => did,
            ty::Dynamic(ref obj, ..) => obj.principal().def_id(),
            ty::Projection(ref proj) => proj.trait_ref(self.tcx).def_id,
            _ => return Some(AccessLevel::Public)
        };
        if let Some(node_id) = self.tcx.hir().as_local_node_id(ty_def_id) {
            self.get(node_id)
        } else {
            Some(AccessLevel::Public)
        }
    }

    fn impl_trait_level(&self, impl_def_id: DefId) -> Option<AccessLevel> {
        if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_def_id) {
            if let Some(node_id) = self.tcx.hir().as_local_node_id(trait_ref.def_id) {
                return self.get(node_id);
            }
        }
        Some(AccessLevel::Public)
    }

    fn get(&self, id: ast::NodeId) -> Option<AccessLevel> {
        self.access_levels.map.get(&id).cloned()
    }

    // Updates node level and returns the updated level.
    fn update(&mut self, id: ast::NodeId, level: Option<AccessLevel>) -> Option<AccessLevel> {
        let old_level = self.get(id);
        // Accessibility levels can only grow.
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
            access_level: self.prev_level.map(|l| l.min(AccessLevel::Reachable)),
            item_def_id: self.tcx.hir().local_def_id(item_id),
            ev: self,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for EmbargoVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let inherited_item_level = match item.node {
            // Impls inherit level from their types and traits.
            hir::ItemKind::Impl(..) => {
                let def_id = self.tcx.hir().local_def_id(item.id);
                cmp::min(self.item_ty_level(def_id), self.impl_trait_level(def_id))
            }
            // Foreign modules inherit level from parents.
            hir::ItemKind::ForeignMod(..) => {
                self.prev_level
            }
            // Other `pub` items inherit levels from parents.
            hir::ItemKind::Const(..) | hir::ItemKind::Enum(..) | hir::ItemKind::ExternCrate(..) |
            hir::ItemKind::GlobalAsm(..) | hir::ItemKind::Fn(..) | hir::ItemKind::Mod(..) |
            hir::ItemKind::Static(..) | hir::ItemKind::Struct(..) |
            hir::ItemKind::Trait(..) | hir::ItemKind::TraitAlias(..) |
            hir::ItemKind::Existential(..) |
            hir::ItemKind::Ty(..) | hir::ItemKind::Union(..) | hir::ItemKind::Use(..) => {
                if item.vis.node.is_pub() { self.prev_level } else { None }
            }
        };

        // Update level of the item itself.
        let item_level = self.update(item.id, inherited_item_level);

        // Update levels of nested things.
        match item.node {
            hir::ItemKind::Enum(ref def, _) => {
                for variant in &def.variants {
                    let variant_level = self.update(variant.node.data.id(), item_level);
                    for field in variant.node.data.fields() {
                        self.update(field.id, variant_level);
                    }
                }
            }
            hir::ItemKind::Impl(.., None, _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    if impl_item_ref.vis.node.is_pub() {
                        self.update(impl_item_ref.id.node_id, item_level);
                    }
                }
            }
            hir::ItemKind::Impl(.., Some(_), _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    self.update(impl_item_ref.id.node_id, item_level);
                }
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                for trait_item_ref in trait_item_refs {
                    self.update(trait_item_ref.id.node_id, item_level);
                }
            }
            hir::ItemKind::Struct(ref def, _) | hir::ItemKind::Union(ref def, _) => {
                if !def.is_struct() {
                    self.update(def.id(), item_level);
                }
                for field in def.fields() {
                    if field.vis.node.is_pub() {
                        self.update(field.id, item_level);
                    }
                }
            }
            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if foreign_item.vis.node.is_pub() {
                        self.update(foreign_item.id, item_level);
                    }
                }
            }
            // Impl trait return types mark their parent function.
            // It (and its children) are revisited if the change applies.
            hir::ItemKind::Existential(ref ty_data) => {
                if let Some(impl_trait_fn) = ty_data.impl_trait_fn {
                    if let Some(node_id) = self.tcx.hir().as_local_node_id(impl_trait_fn) {
                        self.update(node_id, Some(AccessLevel::ReachableFromImplTrait));
                    }
                }
            }
            hir::ItemKind::Use(..) |
            hir::ItemKind::Static(..) |
            hir::ItemKind::Const(..) |
            hir::ItemKind::GlobalAsm(..) |
            hir::ItemKind::Ty(..) |
            hir::ItemKind::Mod(..) |
            hir::ItemKind::TraitAlias(..) |
            hir::ItemKind::Fn(..) |
            hir::ItemKind::ExternCrate(..) => {}
        }

        // Store this node's access level here to propagate the correct
        // reachability level through interfaces and children.
        let orig_level = replace(&mut self.prev_level, item_level);

        // Mark all items in interfaces of reachable items as reachable.
        match item.node {
            // The interface is empty.
            hir::ItemKind::ExternCrate(..) => {}
            // All nested items are checked by `visit_item`.
            hir::ItemKind::Mod(..) => {}
            // Re-exports are handled in `visit_mod`.
            hir::ItemKind::Use(..) => {}
            // The interface is empty.
            hir::ItemKind::GlobalAsm(..) => {}
            hir::ItemKind::Existential(hir::ExistTy { impl_trait_fn: Some(_), .. }) => {
                if item_level.is_some() {
                    // Reach the (potentially private) type and the API being exposed.
                    self.reach(item.id).ty().predicates();
                }
            }
            // Visit everything.
            hir::ItemKind::Const(..) | hir::ItemKind::Static(..) |
            hir::ItemKind::Existential(..) |
            hir::ItemKind::Fn(..) | hir::ItemKind::Ty(..) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates().ty();
                }
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();

                    for trait_item_ref in trait_item_refs {
                        let mut reach = self.reach(trait_item_ref.id.node_id);
                        reach.generics().predicates();

                        if trait_item_ref.kind == hir::AssociatedItemKind::Type &&
                           !trait_item_ref.defaultness.has_value() {
                            // No type to visit.
                        } else {
                            reach.ty();
                        }
                    }
                }
            }
            hir::ItemKind::TraitAlias(..) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();
                }
            }
            // Visit everything except for private impl items.
            hir::ItemKind::Impl(.., ref trait_ref, _, ref impl_item_refs) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates().impl_trait_ref();

                    for impl_item_ref in impl_item_refs {
                        let id = impl_item_ref.id.node_id;
                        if trait_ref.is_some() || self.get(id).is_some() {
                            self.reach(id).generics().predicates().ty();
                        }
                    }
                }
            }

            // Visit everything, but enum variants have their own levels.
            hir::ItemKind::Enum(ref def, _) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();
                }
                for variant in &def.variants {
                    if self.get(variant.node.data.id()).is_some() {
                        for field in variant.node.data.fields() {
                            self.reach(field.id).ty();
                        }
                        // Corner case: if the variant is reachable, but its
                        // enum is not, make the enum reachable as well.
                        self.update(item.id, Some(AccessLevel::Reachable));
                    }
                }
            }
            // Visit everything, but foreign items have their own levels.
            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if self.get(foreign_item.id).is_some() {
                        self.reach(foreign_item.id).generics().predicates().ty();
                    }
                }
            }
            // Visit everything except for private fields.
            hir::ItemKind::Struct(ref struct_def, _) |
            hir::ItemKind::Union(ref struct_def, _) => {
                if item_level.is_some() {
                    self.reach(item.id).generics().predicates();
                    for field in struct_def.fields() {
                        if self.get(field.id).is_some() {
                            self.reach(field.id).ty();
                        }
                    }
                }
            }
        }

        intravisit::walk_item(self, item);

        self.prev_level = orig_level;
    }

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        let orig_level = replace(&mut self.prev_level, None);

        // Blocks can have public items, for example impls, but they always
        // start as completely private regardless of publicity of a function,
        // constant, type, field, etc., in which this block resides.
        intravisit::walk_block(self, b);

        self.prev_level = orig_level;
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod, _sp: Span, id: ast::NodeId) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_level.is_some() {
            let def_id = self.tcx.hir().local_def_id(id);
            if let Some(exports) = self.tcx.module_exports(def_id) {
                for export in exports.iter() {
                    if export.vis == ty::Visibility::Public {
                        if let Some(def_id) = export.def.opt_def_id() {
                            if let Some(node_id) = self.tcx.hir().as_local_node_id(def_id) {
                                self.update(node_id, Some(AccessLevel::Exported));
                            }
                        }
                    }
                }
            }
        }

        intravisit::walk_mod(self, m, id);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
        if md.legacy {
            self.update(md.id, Some(AccessLevel::Public));
            return
        }

        let module_did = ty::DefIdTree::parent(
            self.tcx,
            self.tcx.hir().local_def_id(md.id)
        ).unwrap();
        let mut module_id = self.tcx.hir().as_local_node_id(module_did).unwrap();
        let level = if md.vis.node.is_pub() { self.get(module_id) } else { None };
        let level = self.update(md.id, level);
        if level.is_none() {
            return
        }

        loop {
            let module = if module_id == ast::CRATE_NODE_ID {
                &self.tcx.hir().krate().module
            } else if let hir::ItemKind::Mod(ref module) =
                          self.tcx.hir().expect_item(module_id).node {
                module
            } else {
                unreachable!()
            };
            for id in &module.item_ids {
                self.update(id.id, level);
            }
            let def_id = self.tcx.hir().local_def_id(module_id);
            if let Some(exports) = self.tcx.module_exports(def_id) {
                for export in exports.iter() {
                    if let Some(node_id) = self.tcx.hir().as_local_node_id(export.def.def_id()) {
                        self.update(node_id, level);
                    }
                }
            }

            if module_id == ast::CRATE_NODE_ID {
                break
            }
            module_id = self.tcx.hir().get_parent_node(module_id);
        }
    }
}

impl<'b, 'a, 'tcx> ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
    fn generics(&mut self) -> &mut Self {
        for param in &self.ev.tcx.generics_of(self.item_def_id).params {
            match param.kind {
                GenericParamDefKind::Type { has_default, .. } => {
                    if has_default {
                        self.ev.tcx.type_of(param.def_id).visit_with(self);
                    }
                }
                GenericParamDefKind::Lifetime => {}
            }
        }
        self
    }

    fn predicates(&mut self) -> &mut Self {
        let predicates = self.ev.tcx.predicates_of(self.item_def_id);
        for (predicate, _) in &predicates.predicates {
            predicate.visit_with(self);
            match predicate {
                &ty::Predicate::Trait(poly_predicate) => {
                    self.check_trait_ref(poly_predicate.skip_binder().trait_ref);
                },
                &ty::Predicate::Projection(poly_predicate) => {
                    let tcx = self.ev.tcx;
                    self.check_trait_ref(
                        poly_predicate.skip_binder().projection_ty.trait_ref(tcx)
                    );
                },
                _ => (),
            };
        }
        self
    }

    fn ty(&mut self) -> &mut Self {
        let ty = self.ev.tcx.type_of(self.item_def_id);
        ty.visit_with(self);
        if let ty::FnDef(def_id, _) = ty.sty {
            if def_id == self.item_def_id {
                self.ev.tcx.fn_sig(def_id).visit_with(self);
            }
        }
        self
    }

    fn impl_trait_ref(&mut self) -> &mut Self {
        if let Some(impl_trait_ref) = self.ev.tcx.impl_trait_ref(self.item_def_id) {
            self.check_trait_ref(impl_trait_ref);
            impl_trait_ref.super_visit_with(self);
        }
        self
    }

    fn check_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>) {
        if let Some(node_id) = self.ev.tcx.hir().as_local_node_id(trait_ref.def_id) {
            let item = self.ev.tcx.hir().expect_item(node_id);
            self.ev.update(item.id, self.access_level);
        }
    }
}

impl<'b, 'a, 'tcx> TypeVisitor<'tcx> for ReachEverythingInTheInterfaceVisitor<'b, 'a, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        let ty_def_id = match ty.sty {
            ty::Adt(adt, _) => Some(adt.did),
            ty::Foreign(did) => Some(did),
            ty::Dynamic(ref obj, ..) => Some(obj.principal().def_id()),
            ty::Projection(ref proj) => Some(proj.item_def_id),
            ty::FnDef(def_id, ..) |
            ty::Closure(def_id, ..) |
            ty::Generator(def_id, ..) |
            ty::Opaque(def_id, _) => Some(def_id),
            _ => None
        };

        if let Some(def_id) = ty_def_id {
            if let Some(node_id) = self.ev.tcx.hir().as_local_node_id(def_id) {
                self.ev.update(node_id, self.access_level);
            }
        }

        ty.super_visit_with(self)
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Name privacy visitor, checks privacy and reports violations.
/// Most of name privacy checks are performed during the main resolution phase,
/// or later in type checking when field accesses and associated items are resolved.
/// This pass performs remaining checks for fields in struct expressions and patterns.
//////////////////////////////////////////////////////////////////////////////////////

struct NamePrivacyVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    current_item: ast::NodeId,
    empty_tables: &'a ty::TypeckTables<'tcx>,
}

impl<'a, 'tcx> NamePrivacyVisitor<'a, 'tcx> {
    // Checks that a field in a struct constructor (expression or pattern) is accessible.
    fn check_field(&mut self,
                   use_ctxt: Span, // syntax context of the field name at the use site
                   span: Span, // span of the field pattern, e.g., `x: 0`
                   def: &'tcx ty::AdtDef, // definition of the struct or enum
                   field: &'tcx ty::FieldDef) { // definition of the field
        let ident = Ident::new(keywords::Invalid.name(), use_ctxt);
        let def_id = self.tcx.adjust_ident(ident, def.did, self.current_item).1;
        if !def.is_enum() && !field.vis.is_accessible_from(def_id, self.tcx) {
            struct_span_err!(self.tcx.sess, span, E0451, "field `{}` of {} `{}` is private",
                             field.ident, def.variant_descr(), self.tcx.item_path_str(def.did))
                .span_label(span, format!("field `{}` is private", field.ident))
                .emit();
        }
    }
}

// Set the correct `TypeckTables` for the given `item_id` (or an empty table if
// there is no `TypeckTables` for the item).
fn update_tables<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           item_id: ast::NodeId,
                           tables: &mut &'a ty::TypeckTables<'tcx>,
                           empty_tables: &'a ty::TypeckTables<'tcx>)
                           -> &'a ty::TypeckTables<'tcx> {
    let def_id = tcx.hir().local_def_id(item_id);

    if tcx.has_typeck_tables(def_id) {
        replace(tables, tcx.typeck_tables_of(def_id))
    } else {
        replace(tables, empty_tables)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for NamePrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let orig_tables = replace(&mut self.tables, self.tcx.body_tables(body));
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = orig_tables;
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let orig_current_item = replace(&mut self.current_item, item.id);
        let orig_tables = update_tables(self.tcx, item.id, &mut self.tables, self.empty_tables);
        intravisit::walk_item(self, item);
        self.current_item = orig_current_item;
        self.tables = orig_tables;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        let orig_tables = update_tables(self.tcx, ti.id, &mut self.tables, self.empty_tables);
        intravisit::walk_trait_item(self, ti);
        self.tables = orig_tables;
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let orig_tables = update_tables(self.tcx, ii.id, &mut self.tables, self.empty_tables);
        intravisit::walk_impl_item(self, ii);
        self.tables = orig_tables;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprKind::Struct(ref qpath, ref fields, ref base) => {
                let def = self.tables.qpath_def(qpath, expr.hir_id);
                let adt = self.tables.expr_ty(expr).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(def);
                if let Some(ref base) = *base {
                    // If the expression uses FRU we need to make sure all the unmentioned fields
                    // are checked for privacy (RFC 736). Rather than computing the set of
                    // unmentioned fields, just check them all.
                    for (vf_index, variant_field) in variant.fields.iter().enumerate() {
                        let field = fields.iter().find(|f| {
                            self.tcx.field_index(f.id, self.tables) == vf_index
                        });
                        let (use_ctxt, span) = match field {
                            Some(field) => (field.ident.span, field.span),
                            None => (base.span, base.span),
                        };
                        self.check_field(use_ctxt, span, adt, variant_field);
                    }
                } else {
                    for field in fields {
                        let use_ctxt = field.ident.span;
                        let index = self.tcx.field_index(field.id, self.tables);
                        self.check_field(use_ctxt, field.span, adt, &variant.fields[index]);
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat) {
        match pat.node {
            PatKind::Struct(ref qpath, ref fields, _) => {
                let def = self.tables.qpath_def(qpath, pat.hir_id);
                let adt = self.tables.pat_ty(pat).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(def);
                for field in fields {
                    let use_ctxt = field.node.ident.span;
                    let index = self.tcx.field_index(field.node.id, self.tables);
                    self.check_field(use_ctxt, field.span, adt, &variant.fields[index]);
                }
            }
            _ => {}
        }

        intravisit::walk_pat(self, pat);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Type privacy visitor, checks types for privacy and reports violations.
/// Both explicitly written types and inferred types of expressions and patters are checked.
/// Checks are performed on "semantic" types regardless of names and their hygiene.
////////////////////////////////////////////////////////////////////////////////////////////

struct TypePrivacyVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    current_item: DefId,
    in_body: bool,
    span: Span,
    empty_tables: &'a ty::TypeckTables<'tcx>,
    visited_opaque_tys: FxHashSet<DefId>
}

impl<'a, 'tcx> TypePrivacyVisitor<'a, 'tcx> {
    fn def_id_visibility(&self, did: DefId) -> ty::Visibility {
        match self.tcx.hir().as_local_node_id(did) {
            Some(node_id) => {
                let vis = match self.tcx.hir().get(node_id) {
                    Node::Item(item) => &item.vis,
                    Node::ForeignItem(foreign_item) => &foreign_item.vis,
                    Node::ImplItem(impl_item) => &impl_item.vis,
                    Node::TraitItem(..) |
                    Node::Variant(..) => {
                        return self.def_id_visibility(self.tcx.hir().get_parent_did(node_id));
                    }
                    Node::StructCtor(vdata) => {
                        let struct_node_id = self.tcx.hir().get_parent(node_id);
                        let struct_vis = match self.tcx.hir().get(struct_node_id) {
                            Node::Item(item) => &item.vis,
                            node => bug!("unexpected node kind: {:?}", node),
                        };
                        let mut ctor_vis
                            = ty::Visibility::from_hir(struct_vis, struct_node_id, self.tcx);
                        for field in vdata.fields() {
                            let field_vis = ty::Visibility::from_hir(&field.vis, node_id, self.tcx);
                            if ctor_vis.is_at_least(field_vis, self.tcx) {
                                ctor_vis = field_vis;
                            }
                        }

                        // If the structure is marked as non_exhaustive then lower the
                        // visibility to within the crate.
                        let struct_def_id = self.tcx.hir().get_parent_did(node_id);
                        let adt_def = self.tcx.adt_def(struct_def_id);
                        if adt_def.non_enum_variant().is_field_list_non_exhaustive()
                            && ctor_vis == ty::Visibility::Public
                        {
                            ctor_vis = ty::Visibility::Restricted(
                                DefId::local(CRATE_DEF_INDEX));
                        }

                        return ctor_vis;
                    }
                    node => bug!("unexpected node kind: {:?}", node)
                };
                ty::Visibility::from_hir(vis, node_id, self.tcx)
            }
            None => self.tcx.visibility(did),
        }
    }

    fn item_is_accessible(&self, did: DefId) -> bool {
        self.def_id_visibility(did).is_accessible_from(self.current_item, self.tcx)
    }

    // Take node-id of an expression or pattern and check its type for privacy.
    fn check_expr_pat_type(&mut self, id: hir::HirId, span: Span) -> bool {
        self.span = span;
        if self.tables.node_id_to_type(id).visit_with(self) {
            return true;
        }
        if self.tables.node_substs(id).visit_with(self) {
            return true;
        }
        if let Some(adjustments) = self.tables.adjustments().get(id) {
            for adjustment in adjustments {
                if adjustment.target.visit_with(self) {
                    return true;
                }
            }
        }
        false
    }

    fn check_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>) -> bool {
        if !self.item_is_accessible(trait_ref.def_id) {
            let msg = format!("trait `{}` is private", trait_ref);
            self.tcx.sess.span_err(self.span, &msg);
            return true;
        }

        trait_ref.super_visit_with(self)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypePrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let orig_tables = replace(&mut self.tables, self.tcx.body_tables(body));
        let orig_in_body = replace(&mut self.in_body, true);
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = orig_tables;
        self.in_body = orig_in_body;
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty) {
        self.span = hir_ty.span;
        if self.in_body {
            // Types in bodies.
            if self.tables.node_id_to_type(hir_ty.hir_id).visit_with(self) {
                return;
            }
        } else {
            // Types in signatures.
            // FIXME: This is very ineffective. Ideally each HIR type should be converted
            // into a semantic type only once and the result should be cached somehow.
            if rustc_typeck::hir_ty_to_ty(self.tcx, hir_ty).visit_with(self) {
                return;
            }
        }

        intravisit::walk_ty(self, hir_ty);
    }

    fn visit_trait_ref(&mut self, trait_ref: &'tcx hir::TraitRef) {
        self.span = trait_ref.path.span;
        if !self.in_body {
            // Avoid calling `hir_trait_to_predicates` in bodies, it will ICE.
            // The traits' privacy in bodies is already checked as a part of trait object types.
            let (principal, projections) =
                rustc_typeck::hir_trait_to_predicates(self.tcx, trait_ref);
            if self.check_trait_ref(*principal.skip_binder()) {
                return;
            }
            for (poly_predicate, _) in projections {
                let tcx = self.tcx;
                if self.check_trait_ref(poly_predicate.skip_binder().projection_ty.trait_ref(tcx)) {
                    return;
                }
            }
        }

        intravisit::walk_trait_ref(self, trait_ref);
    }

    // Check types of expressions
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if self.check_expr_pat_type(expr.hir_id, expr.span) {
            // Do not check nested expressions if the error already happened.
            return;
        }
        match expr.node {
            hir::ExprKind::Assign(.., ref rhs) | hir::ExprKind::Match(ref rhs, ..) => {
                // Do not report duplicate errors for `x = y` and `match x { ... }`.
                if self.check_expr_pat_type(rhs.hir_id, rhs.span) {
                    return;
                }
            }
            hir::ExprKind::MethodCall(_, span, _) => {
                // Method calls have to be checked specially.
                self.span = span;
                if let Some(def) = self.tables.type_dependent_defs().get(expr.hir_id) {
                    let def_id = def.def_id();
                    if self.tcx.type_of(def_id).visit_with(self) {
                        return;
                    }
                } else {
                    self.tcx.sess.delay_span_bug(expr.span,
                                                 "no type-dependent def for method call");
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    // Prohibit access to associated items with insufficient nominal visibility.
    //
    // Additionally, until better reachability analysis for macros 2.0 is available,
    // we prohibit access to private statics from other crates, this allows to give
    // more code internal visibility at link time. (Access to private functions
    // is already prohibited by type privacy for function types.)
    fn visit_qpath(&mut self, qpath: &'tcx hir::QPath, id: hir::HirId, span: Span) {
        let def = match *qpath {
            hir::QPath::Resolved(_, ref path) => match path.def {
                Def::Method(..) | Def::AssociatedConst(..) |
                Def::AssociatedTy(..) | Def::Static(..) => Some(path.def),
                _ => None,
            }
            hir::QPath::TypeRelative(..) => {
                self.tables.type_dependent_defs().get(id).cloned()
            }
        };
        if let Some(def) = def {
            let def_id = def.def_id();
            let is_local_static = if let Def::Static(..) = def { def_id.is_local() } else { false };
            if !self.item_is_accessible(def_id) && !is_local_static {
                let name = match *qpath {
                    hir::QPath::Resolved(_, ref path) => path.to_string(),
                    hir::QPath::TypeRelative(_, ref segment) => segment.ident.to_string(),
                };
                let msg = format!("{} `{}` is private", def.kind_name(), name);
                self.tcx.sess.span_err(span, &msg);
                return;
            }
        }

        intravisit::walk_qpath(self, qpath, id, span);
    }

    // Check types of patterns.
    fn visit_pat(&mut self, pattern: &'tcx hir::Pat) {
        if self.check_expr_pat_type(pattern.hir_id, pattern.span) {
            // Do not check nested patterns if the error already happened.
            return;
        }

        intravisit::walk_pat(self, pattern);
    }

    fn visit_local(&mut self, local: &'tcx hir::Local) {
        if let Some(ref init) = local.init {
            if self.check_expr_pat_type(init.hir_id, init.span) {
                // Do not report duplicate errors for `let x = y`.
                return;
            }
        }

        intravisit::walk_local(self, local);
    }

    // Check types in item interfaces.
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let orig_current_item = self.current_item;
        let orig_tables = update_tables(self.tcx,
                                        item.id,
                                        &mut self.tables,
                                        self.empty_tables);
        let orig_in_body = replace(&mut self.in_body, false);
        self.current_item = self.tcx.hir().local_def_id(item.id);
        intravisit::walk_item(self, item);
        self.tables = orig_tables;
        self.in_body = orig_in_body;
        self.current_item = orig_current_item;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        let orig_tables = update_tables(self.tcx, ti.id, &mut self.tables, self.empty_tables);
        intravisit::walk_trait_item(self, ti);
        self.tables = orig_tables;
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let orig_tables = update_tables(self.tcx, ii.id, &mut self.tables, self.empty_tables);
        intravisit::walk_impl_item(self, ii);
        self.tables = orig_tables;
    }
}

impl<'a, 'tcx> TypeVisitor<'tcx> for TypePrivacyVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::Adt(&ty::AdtDef { did: def_id, .. }, ..) |
            ty::FnDef(def_id, ..) |
            ty::Foreign(def_id) => {
                if !self.item_is_accessible(def_id) {
                    let msg = format!("type `{}` is private", ty);
                    self.tcx.sess.span_err(self.span, &msg);
                    return true;
                }
                if let ty::FnDef(..) = ty.sty {
                    if self.tcx.fn_sig(def_id).visit_with(self) {
                        return true;
                    }
                }
                // Inherent static methods don't have self type in substs,
                // we have to check it additionally.
                if let Some(assoc_item) = self.tcx.opt_associated_item(def_id) {
                    if let ty::ImplContainer(impl_def_id) = assoc_item.container {
                        if self.tcx.type_of(impl_def_id).visit_with(self) {
                            return true;
                        }
                    }
                }
            }
            ty::Dynamic(ref predicates, ..) => {
                let is_private = predicates.skip_binder().iter().any(|predicate| {
                    let def_id = match *predicate {
                        ty::ExistentialPredicate::Trait(trait_ref) => trait_ref.def_id,
                        ty::ExistentialPredicate::Projection(proj) =>
                            proj.trait_ref(self.tcx).def_id,
                        ty::ExistentialPredicate::AutoTrait(def_id) => def_id,
                    };
                    !self.item_is_accessible(def_id)
                });
                if is_private {
                    let msg = format!("type `{}` is private", ty);
                    self.tcx.sess.span_err(self.span, &msg);
                    return true;
                }
            }
            ty::Projection(ref proj) => {
                let tcx = self.tcx;
                if self.check_trait_ref(proj.trait_ref(tcx)) {
                    return true;
                }
            }
            ty::Opaque(def_id, ..) => {
                for (predicate, _) in &self.tcx.predicates_of(def_id).predicates {
                    let trait_ref = match *predicate {
                        ty::Predicate::Trait(ref poly_trait_predicate) => {
                            Some(poly_trait_predicate.skip_binder().trait_ref)
                        }
                        ty::Predicate::Projection(ref poly_projection_predicate) => {
                            if poly_projection_predicate.skip_binder().ty.visit_with(self) {
                                return true;
                            }
                            Some(poly_projection_predicate.skip_binder()
                                                          .projection_ty.trait_ref(self.tcx))
                        }
                        ty::Predicate::TypeOutlives(..) | ty::Predicate::RegionOutlives(..) => None,
                        _ => bug!("unexpected predicate: {:?}", predicate),
                    };
                    if let Some(trait_ref) = trait_ref {
                        if !self.item_is_accessible(trait_ref.def_id) {
                            let msg = format!("trait `{}` is private", trait_ref);
                            self.tcx.sess.span_err(self.span, &msg);
                            return true;
                        }
                        for subst in trait_ref.substs.iter() {
                            // Skip repeated `Opaque`s to avoid infinite recursion.
                            if let UnpackedKind::Type(ty) = subst.unpack() {
                                if let ty::Opaque(def_id, ..) = ty.sty {
                                    if !self.visited_opaque_tys.insert(def_id) {
                                        continue;
                                    }
                                }
                            }
                            if subst.visit_with(self) {
                                return true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        ty.super_visit_with(self)
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
    // Set of errors produced by this obsolete visitor.
    old_error_set: NodeSet,
}

struct ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b: 'a, 'tcx: 'b> {
    inner: &'a ObsoleteVisiblePrivateTypesVisitor<'b, 'tcx>,
    /// Whether the type refers to private types.
    contains_private: bool,
    /// Whether we've recurred at all (i.e., if we're pointing at the
    /// first type on which `visit_ty` was called).
    at_outer_type: bool,
    /// Whether that first type is a public path.
    outer_type_is_public_path: bool,
}

impl<'a, 'tcx> ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    fn path_is_private_type(&self, path: &hir::Path) -> bool {
        let did = match path.def {
            Def::PrimTy(..) | Def::SelfTy(..) | Def::Err => return false,
            def => def.def_id(),
        };

        // A path can only be private if:
        // it's in this crate...
        if let Some(node_id) = self.tcx.hir().as_local_node_id(did) {
            // .. and it corresponds to a private type in the AST (this returns
            // `None` for type parameters).
            match self.tcx.hir().find(node_id) {
                Some(Node::Item(ref item)) => !item.vis.node.is_pub(),
                Some(_) | None => false,
            }
        } else {
            return false
        }
    }

    fn trait_is_public(&self, trait_id: ast::NodeId) -> bool {
        // FIXME: this would preferably be using `exported_items`, but all
        // traits are exported currently (see `EmbargoVisitor.exported_trait`).
        self.access_levels.is_public(trait_id)
    }

    fn check_generic_bound(&mut self, bound: &hir::GenericBound) {
        if let hir::GenericBound::Trait(ref trait_ref, _) = *bound {
            if self.path_is_private_type(&trait_ref.trait_ref.path) {
                self.old_error_set.insert(trait_ref.trait_ref.ref_id);
            }
        }
    }

    fn item_is_public(&self, id: &ast::NodeId, vis: &hir::Visibility) -> bool {
        self.access_levels.is_reachable(*id) || vis.node.is_pub()
    }
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

    fn visit_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = ty.node {
            if self.inner.path_is_private_type(path) {
                self.contains_private = true;
                // Found what we're looking for, so let's stop working.
                return
            }
        }
        if let hir::TyKind::Path(_) = ty.node {
            if self.at_outer_type {
                self.outer_type_is_public_path = true;
            }
        }
        self.at_outer_type = false;
        intravisit::walk_ty(self, ty)
    }

    // Don't want to recurse into `[, .. expr]`.
    fn visit_expr(&mut self, _: &hir::Expr) {}
}

impl<'a, 'tcx> Visitor<'tcx> for ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            // Contents of a private mod can be re-exported, so we need
            // to check internals.
            hir::ItemKind::Mod(_) => {}

            // An `extern {}` doesn't introduce a new privacy
            // namespace (the contents have their own privacies).
            hir::ItemKind::ForeignMod(_) => {}

            hir::ItemKind::Trait(.., ref bounds, _) => {
                if !self.trait_is_public(item.id) {
                    return
                }

                for bound in bounds.iter() {
                    self.check_generic_bound(bound)
                }
            }

            // Impls need some special handling to try to offer useful
            // error messages without (too many) false positives
            // (i.e., we could just return here to not check them at
            // all, or some worse estimation of whether an impl is
            // publicly visible).
            hir::ItemKind::Impl(.., ref g, ref trait_ref, ref self_, ref impl_item_refs) => {
                // `impl [... for] Private` is never visible.
                let self_contains_private;
                // `impl [... for] Public<...>`, but not `impl [... for]
                // Vec<Public>` or `(Public,)`, etc.
                let self_is_public_path;

                // Check the properties of the `Self` type:
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

                // Miscellaneous info about the impl:

                // `true` iff this is `impl Private for ...`.
                let not_private_trait =
                    trait_ref.as_ref().map_or(true, // no trait counts as public trait
                                              |tr| {
                        let did = tr.path.def.def_id();

                        if let Some(node_id) = self.tcx.hir().as_local_node_id(did) {
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
                                     let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                                     match impl_item.node {
                                         hir::ImplItemKind::Const(..) |
                                         hir::ImplItemKind::Method(..) => {
                                             self.access_levels.is_reachable(impl_item.id)
                                         }
                                         hir::ImplItemKind::Existential(..) |
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
                                let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
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
                                let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                                if let hir::ImplItemKind::Type(ref ty) = impl_item.node {
                                    self.visit_ty(ty);
                                }
                            }
                        }
                    }
                } else if trait_ref.is_none() && self_is_public_path {
                    // `impl Public<Private> { ... }`. Any public static
                    // methods will be visible as `Public::foo`.
                    let mut found_pub_static = false;
                    for impl_item_ref in impl_item_refs {
                        if self.item_is_public(&impl_item_ref.id.node_id, &impl_item_ref.vis) {
                            let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
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
            hir::ItemKind::Ty(..) => return,

            // Not at all public, so we don't care.
            _ if !self.item_is_public(&item.id, &item.vis) => {
                return;
            }

            _ => {}
        }

        // We've carefully constructed it so that if we're here, then
        // any `visit_ty`'s will be called on things that are in
        // public signatures, i.e., things that we're interested in for
        // this visitor.
        intravisit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
        for param in &generics.params {
            for bound in &param.bounds {
                self.check_generic_bound(bound);
            }
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &hir::WherePredicate::BoundPredicate(ref bound_pred) => {
                    for bound in bound_pred.bounds.iter() {
                        self.check_generic_bound(bound)
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
        if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = t.node {
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
        if s.vis.node.is_pub() || self.in_variant {
            intravisit::walk_struct_field(self, s);
        }
    }

    // We don't need to introspect into these at all: an
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
    /// The visitor checks that each component type is at least this visible.
    required_visibility: ty::Visibility,
    /// The visibility of the least visible component that has been visited.
    min_visibility: ty::Visibility,
    has_pub_restricted: bool,
    has_old_errors: bool,
    in_assoc_ty: bool,
}

impl<'a, 'tcx: 'a> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn generics(&mut self) -> &mut Self {
        for param in &self.tcx.generics_of(self.item_def_id).params {
            match param.kind {
                GenericParamDefKind::Type { has_default, .. } => {
                    if has_default {
                        self.tcx.type_of(param.def_id).visit_with(self);
                    }
                }
                GenericParamDefKind::Lifetime => {}
            }
        }
        self
    }

    fn predicates(&mut self) -> &mut Self {
        // N.B., we use `explicit_predicates_of` and not `predicates_of`
        // because we don't want to report privacy errors due to where
        // clauses that the compiler inferred. We only want to
        // consider the ones that the user wrote. This is important
        // for the inferred outlives rules; see
        // `src/test/ui/rfc-2093-infer-outlives/privacy.rs`.
        let predicates = self.tcx.explicit_predicates_of(self.item_def_id);
        for (predicate, _) in &predicates.predicates {
            predicate.visit_with(self);
            match predicate {
                &ty::Predicate::Trait(poly_predicate) => {
                    self.check_trait_ref(poly_predicate.skip_binder().trait_ref);
                },
                &ty::Predicate::Projection(poly_predicate) => {
                    let tcx = self.tcx;
                    self.check_trait_ref(
                        poly_predicate.skip_binder().projection_ty.trait_ref(tcx)
                    );
                },
                _ => (),
            };
        }
        self
    }

    fn ty(&mut self) -> &mut Self {
        let ty = self.tcx.type_of(self.item_def_id);
        ty.visit_with(self);
        if let ty::FnDef(def_id, _) = ty.sty {
            if def_id == self.item_def_id {
                self.tcx.fn_sig(def_id).visit_with(self);
            }
        }
        self
    }

    fn impl_trait_ref(&mut self) -> &mut Self {
        if let Some(impl_trait_ref) = self.tcx.impl_trait_ref(self.item_def_id) {
            self.check_trait_ref(impl_trait_ref);
            impl_trait_ref.super_visit_with(self);
        }
        self
    }

    fn check_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>) {
        // Non-local means public (private items can't leave their crate, modulo bugs).
        if let Some(node_id) = self.tcx.hir().as_local_node_id(trait_ref.def_id) {
            let item = self.tcx.hir().expect_item(node_id);
            let vis = ty::Visibility::from_hir(&item.vis, node_id, self.tcx);
            if !vis.is_at_least(self.min_visibility, self.tcx) {
                self.min_visibility = vis;
            }
            if !vis.is_at_least(self.required_visibility, self.tcx) {
                if self.has_pub_restricted || self.has_old_errors || self.in_assoc_ty {
                    struct_span_err!(self.tcx.sess, self.span, E0445,
                                     "private trait `{}` in public interface", trait_ref)
                        .span_label(self.span, format!(
                                    "can't leak private trait"))
                        .emit();
                } else {
                    self.tcx.lint_node(lint::builtin::PRIVATE_IN_PUBLIC,
                                       node_id,
                                       self.span,
                                       &format!("private trait `{}` in public \
                                                 interface (error E0445)", trait_ref));
                }
            }
        }
    }
}

impl<'a, 'tcx: 'a> TypeVisitor<'tcx> for SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        let ty_def_id = match ty.sty {
            ty::Adt(adt, _) => Some(adt.did),
            ty::Foreign(did) => Some(did),
            ty::Dynamic(ref obj, ..) => Some(obj.principal().def_id()),
            ty::Projection(ref proj) => {
                if self.required_visibility == ty::Visibility::Invisible {
                    // Conservatively approximate the whole type alias as public without
                    // recursing into its components when determining impl publicity.
                    // For example, `impl <Type as Trait>::Alias {...}` may be a public impl
                    // even if both `Type` and `Trait` are private.
                    // Ideally, associated types should be substituted in the same way as
                    // free type aliases, but this isn't done yet.
                    return false;
                }
                let trait_ref = proj.trait_ref(self.tcx);
                Some(trait_ref.def_id)
            }
            _ => None
        };

        if let Some(def_id) = ty_def_id {
            // Non-local means public (private items can't leave their crate, modulo bugs).
            if let Some(node_id) = self.tcx.hir().as_local_node_id(def_id) {
                let hir_vis = match self.tcx.hir().find(node_id) {
                    Some(Node::Item(item)) => &item.vis,
                    Some(Node::ForeignItem(item)) => &item.vis,
                    _ => bug!("expected item of foreign item"),
                };

                let vis = ty::Visibility::from_hir(hir_vis, node_id, self.tcx);

                if !vis.is_at_least(self.min_visibility, self.tcx) {
                    self.min_visibility = vis;
                }
                if !vis.is_at_least(self.required_visibility, self.tcx) {
                    let vis_adj = match hir_vis.node {
                        hir::VisibilityKind::Crate(_) => "crate-visible",
                        hir::VisibilityKind::Restricted { .. } => "restricted",
                        _ => "private"
                    };

                    if self.has_pub_restricted || self.has_old_errors || self.in_assoc_ty {
                        let mut err = struct_span_err!(self.tcx.sess, self.span, E0446,
                            "{} type `{}` in public interface", vis_adj, ty);
                        err.span_label(self.span, format!("can't leak {} type", vis_adj));
                        err.span_label(hir_vis.span, format!("`{}` declared as {}", ty, vis_adj));
                        err.emit();
                    } else {
                        self.tcx.lint_node(lint::builtin::PRIVATE_IN_PUBLIC,
                                           node_id,
                                           self.span,
                                           &format!("{} type `{}` in public \
                                                     interface (error E0446)", vis_adj, ty));
                    }
                }
            }
        }

        ty.super_visit_with(self)
    }
}

struct PrivateItemsInPublicInterfacesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    has_pub_restricted: bool,
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
                let parent = self.tcx.hir().get_parent_node(id);
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
            item_def_id: self.tcx.hir().local_def_id(item_id),
            span: self.tcx.hir().span(item_id),
            min_visibility: ty::Visibility::Public,
            required_visibility,
            has_pub_restricted: self.has_pub_restricted,
            has_old_errors,
            in_assoc_ty: false,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let tcx = self.tcx;
        let min = |vis1: ty::Visibility, vis2| {
            if vis1.is_at_least(vis2, tcx) { vis2 } else { vis1 }
        };

        let item_visibility = ty::Visibility::from_hir(&item.vis, item.id, tcx);

        match item.node {
            // Crates are always public.
            hir::ItemKind::ExternCrate(..) => {}
            // All nested items are checked by `visit_item`.
            hir::ItemKind::Mod(..) => {}
            // Checked in resolve.
            hir::ItemKind::Use(..) => {}
            // No subitems.
            hir::ItemKind::GlobalAsm(..) => {}
            hir::ItemKind::Existential(hir::ExistTy { impl_trait_fn: Some(_), .. }) => {
                // Check the traits being exposed, as they're separate,
                // e.g., `impl Iterator<Item=T>` has two predicates,
                // `X: Iterator` and `<X as Iterator>::Item == T`,
                // where `X` is the `impl Iterator<Item=T>` itself,
                // stored in `predicates_of`, not in the `Ty` itself.
                self.check(item.id, item_visibility).predicates();
            }
            // Subitems of these items have inherited publicity.
            hir::ItemKind::Const(..) | hir::ItemKind::Static(..) | hir::ItemKind::Fn(..) |
            hir::ItemKind::Existential(..) |
            hir::ItemKind::Ty(..) => {
                self.check(item.id, item_visibility).generics().predicates().ty();

                // Recurse for e.g., `impl Trait` (see `visit_ty`).
                self.inner_visibility = item_visibility;
                intravisit::walk_item(self, item);
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                self.check(item.id, item_visibility).generics().predicates();

                for trait_item_ref in trait_item_refs {
                    let mut check = self.check(trait_item_ref.id.node_id, item_visibility);
                    check.in_assoc_ty = trait_item_ref.kind == hir::AssociatedItemKind::Type;
                    check.generics().predicates();

                    if trait_item_ref.kind == hir::AssociatedItemKind::Type &&
                       !trait_item_ref.defaultness.has_value() {
                        // No type to visit.
                    } else {
                        check.ty();
                    }
                }
            }
            hir::ItemKind::TraitAlias(..) => {
                self.check(item.id, item_visibility).generics().predicates();
            }
            hir::ItemKind::Enum(ref def, _) => {
                self.check(item.id, item_visibility).generics().predicates();

                for variant in &def.variants {
                    for field in variant.node.data.fields() {
                        self.check(field.id, item_visibility).ty();
                    }
                }
            }
            // Subitems of foreign modules have their own publicity.
            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    let vis = ty::Visibility::from_hir(&foreign_item.vis, item.id, tcx);
                    self.check(foreign_item.id, vis).generics().predicates().ty();
                }
            }
            // Subitems of structs and unions have their own publicity.
            hir::ItemKind::Struct(ref struct_def, _) |
            hir::ItemKind::Union(ref struct_def, _) => {
                self.check(item.id, item_visibility).generics().predicates();

                for field in struct_def.fields() {
                    let field_visibility = ty::Visibility::from_hir(&field.vis, item.id, tcx);
                    self.check(field.id, min(item_visibility, field_visibility)).ty();
                }
            }
            // An inherent impl is public when its type is public
            // Subitems of inherent impls have their own publicity.
            hir::ItemKind::Impl(.., None, _, ref impl_item_refs) => {
                let ty_vis =
                    self.check(item.id, ty::Visibility::Invisible).ty().min_visibility;
                self.check(item.id, ty_vis).generics().predicates();

                for impl_item_ref in impl_item_refs {
                    let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                    let impl_item_vis = ty::Visibility::from_hir(&impl_item.vis, item.id, tcx);
                    let mut check = self.check(impl_item.id, min(impl_item_vis, ty_vis));
                    check.in_assoc_ty = impl_item_ref.kind == hir::AssociatedItemKind::Type;
                    check.generics().predicates().ty();

                    // Recurse for e.g., `impl Trait` (see `visit_ty`).
                    self.inner_visibility = impl_item_vis;
                    intravisit::walk_impl_item(self, impl_item);
                }
            }
            // A trait impl is public when both its type and its trait are public
            // Subitems of trait impls have inherited publicity.
            hir::ItemKind::Impl(.., Some(_), _, ref impl_item_refs) => {
                let vis = self.check(item.id, ty::Visibility::Invisible)
                              .ty().impl_trait_ref().min_visibility;
                self.check(item.id, vis).generics().predicates();
                for impl_item_ref in impl_item_refs {
                    let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                    let mut check = self.check(impl_item.id, vis);
                    check.in_assoc_ty = impl_item_ref.kind == hir::AssociatedItemKind::Type;
                    check.generics().predicates().ty();

                    // Recurse for e.g., `impl Trait` (see `visit_ty`).
                    self.inner_visibility = vis;
                    intravisit::walk_impl_item(self, impl_item);
                }
            }
        }
    }

    fn visit_impl_item(&mut self, _impl_item: &'tcx hir::ImplItem) {
        // Handled in `visit_item` above.
    }

    // Don't recurse into expressions in array sizes or const initializers.
    fn visit_expr(&mut self, _: &'tcx hir::Expr) {}
    // Don't recurse into patterns in function arguments.
    fn visit_pat(&mut self, _: &'tcx hir::Pat) {}
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        privacy_access_levels,
        ..*providers
    };
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Lrc<AccessLevels> {
    tcx.privacy_access_levels(LOCAL_CRATE)
}

fn privacy_access_levels<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                   krate: CrateNum)
                                   -> Lrc<AccessLevels> {
    assert_eq!(krate, LOCAL_CRATE);

    let krate = tcx.hir().krate();
    let empty_tables = ty::TypeckTables::empty(None);

    // Check privacy of names not checked in previous compilation stages.
    let mut visitor = NamePrivacyVisitor {
        tcx,
        tables: &empty_tables,
        current_item: CRATE_NODE_ID,
        empty_tables: &empty_tables,
    };
    intravisit::walk_crate(&mut visitor, krate);

    // Check privacy of explicitly written types and traits as well as
    // inferred types of expressions and patterns.
    let mut visitor = TypePrivacyVisitor {
        tcx,
        tables: &empty_tables,
        current_item: DefId::local(CRATE_DEF_INDEX),
        in_body: false,
        span: krate.span,
        empty_tables: &empty_tables,
        visited_opaque_tys: FxHashSet::default()
    };
    intravisit::walk_crate(&mut visitor, krate);

    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx,
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
            tcx,
            access_levels: &visitor.access_levels,
            in_variant: false,
            old_error_set: Default::default(),
        };
        intravisit::walk_crate(&mut visitor, krate);


        let has_pub_restricted = {
            let mut pub_restricted_visitor = PubRestrictedVisitor {
                tcx,
                has_pub_restricted: false
            };
            intravisit::walk_crate(&mut pub_restricted_visitor, krate);
            pub_restricted_visitor.has_pub_restricted
        };

        // Check for private types and traits in public interfaces.
        let mut visitor = PrivateItemsInPublicInterfacesVisitor {
            tcx,
            has_pub_restricted,
            old_error_set: &visitor.old_error_set,
            inner_visibility: ty::Visibility::Public,
        };
        krate.visit_all_item_likes(&mut DeepVisitor::new(&mut visitor));
    }

    Lrc::new(visitor.access_levels)
}

__build_diagnostic_array! { librustc_privacy, DIAGNOSTICS }
