// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This implements the dead-code warning pass. It follows middle::reachable
// closely. The idea is that all reachable symbols are live, codes called
// from live codes are live, and everything else is dead.

use front::map as ast_map;
use rustc_front::hir;
use rustc_front::visit::{self, Visitor};

use middle::{def, pat_util, privacy, ty};
use middle::def_id::{DefId};
use lint;
use util::nodemap::NodeSet;

use std::collections::HashSet;
use syntax::{ast, codemap};
use syntax::attr::{self, AttrMetaMethods};

// Any local node that may call something in its body block should be
// explored. For example, if it's a live NodeItem that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore(tcx: &ty::ctxt, node_id: ast::NodeId) -> bool {
    match tcx.map.find(node_id) {
        Some(ast_map::NodeItem(..)) |
        Some(ast_map::NodeImplItem(..)) |
        Some(ast_map::NodeForeignItem(..)) |
        Some(ast_map::NodeTraitItem(..)) =>
            true,
        _ =>
            false
    }
}

struct MarkSymbolVisitor<'a, 'tcx: 'a> {
    worklist: Vec<ast::NodeId>,
    tcx: &'a ty::ctxt<'tcx>,
    live_symbols: Box<HashSet<ast::NodeId>>,
    struct_has_extern_repr: bool,
    ignore_non_const_paths: bool,
    inherited_pub_visibility: bool,
    ignore_variant_stack: Vec<DefId>,
}

impl<'a, 'tcx> MarkSymbolVisitor<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           worklist: Vec<ast::NodeId>) -> MarkSymbolVisitor<'a, 'tcx> {
        MarkSymbolVisitor {
            worklist: worklist,
            tcx: tcx,
            live_symbols: box HashSet::new(),
            struct_has_extern_repr: false,
            ignore_non_const_paths: false,
            inherited_pub_visibility: false,
            ignore_variant_stack: vec![],
        }
    }

    fn check_def_id(&mut self, def_id: DefId) {
        if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
            if should_explore(self.tcx, node_id) {
                self.worklist.push(node_id);
            }
            self.live_symbols.insert(node_id);
        }
    }

    fn insert_def_id(&mut self, def_id: DefId) {
        if let Some(node_id) = self.tcx.map.as_local_node_id(def_id) {
            debug_assert!(!should_explore(self.tcx, node_id));
            self.live_symbols.insert(node_id);
        }
    }

    fn lookup_and_handle_definition(&mut self, id: &ast::NodeId) {
        use middle::ty::TypeVariants::{TyEnum, TyStruct};

        // If `bar` is a trait item, make sure to mark Foo as alive in `Foo::bar`
        self.tcx.tables.borrow().item_substs.get(id)
            .and_then(|substs| substs.substs.self_ty())
            .map(|ty| match ty.sty {
                TyEnum(tyid, _) | TyStruct(tyid, _) => self.check_def_id(tyid.did),
                _ => (),
            });

        self.tcx.def_map.borrow().get(id).map(|def| {
            match def.full_def() {
                def::DefConst(_) | def::DefAssociatedConst(..) => {
                    self.check_def_id(def.def_id());
                }
                _ if self.ignore_non_const_paths => (),
                def::DefPrimTy(_) => (),
                def::DefSelfTy(..) => (),
                def::DefVariant(enum_id, variant_id, _) => {
                    self.check_def_id(enum_id);
                    if !self.ignore_variant_stack.contains(&variant_id) {
                        self.check_def_id(variant_id);
                    }
                }
                _ => {
                    self.check_def_id(def.def_id());
                }
            }
        });
    }

    fn lookup_and_handle_method(&mut self, id: ast::NodeId) {
        let method_call = ty::MethodCall::expr(id);
        let method = self.tcx.tables.borrow().method_map[&method_call];
        self.check_def_id(method.def_id);
    }

    fn handle_field_access(&mut self, lhs: &hir::Expr, name: ast::Name) {
        if let ty::TyStruct(def, _) = self.tcx.expr_ty_adjusted(lhs).sty {
            self.insert_def_id(def.struct_variant().field_named(name).did);
        } else {
            self.tcx.sess.span_bug(lhs.span, "named field access on non-struct")
        }
    }

    fn handle_tup_field_access(&mut self, lhs: &hir::Expr, idx: usize) {
        if let ty::TyStruct(def, _) = self.tcx.expr_ty_adjusted(lhs).sty {
            self.insert_def_id(def.struct_variant().fields[idx].did);
        }
    }

    fn handle_field_pattern_match(&mut self, lhs: &hir::Pat,
                                  pats: &[codemap::Spanned<hir::FieldPat>]) {
        let def = self.tcx.def_map.borrow().get(&lhs.id).unwrap().full_def();
        let pat_ty = self.tcx.node_id_to_type(lhs.id);
        let variant = match pat_ty.sty {
            ty::TyStruct(adt, _) | ty::TyEnum(adt, _) => adt.variant_of_def(def),
            _ => self.tcx.sess.span_bug(lhs.span, "non-ADT in struct pattern")
        };
        for pat in pats {
            if let hir::PatWild = pat.node.pat.node {
                continue;
            }
            self.insert_def_id(variant.field_named(pat.node.name).did);
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = HashSet::new();
        while !self.worklist.is_empty() {
            let id = self.worklist.pop().unwrap();
            if scanned.contains(&id) {
                continue
            }
            scanned.insert(id);

            match self.tcx.map.find(id) {
                Some(ref node) => {
                    self.live_symbols.insert(id);
                    self.visit_node(node);
                }
                None => (),
            }
        }
    }

    fn visit_node(&mut self, node: &ast_map::Node) {
        let had_extern_repr = self.struct_has_extern_repr;
        self.struct_has_extern_repr = false;
        let had_inherited_pub_visibility = self.inherited_pub_visibility;
        self.inherited_pub_visibility = false;
        match *node {
            ast_map::NodeItem(item) => {
                match item.node {
                    hir::ItemStruct(..) => {
                        self.struct_has_extern_repr = item.attrs.iter().any(|attr| {
                            attr::find_repr_attrs(self.tcx.sess.diagnostic(), attr)
                                .contains(&attr::ReprExtern)
                        });

                        visit::walk_item(self, &*item);
                    }
                    hir::ItemEnum(..) => {
                        self.inherited_pub_visibility = item.vis == hir::Public;
                        visit::walk_item(self, &*item);
                    }
                    hir::ItemFn(..)
                    | hir::ItemTy(..)
                    | hir::ItemStatic(..)
                    | hir::ItemConst(..) => {
                        visit::walk_item(self, &*item);
                    }
                    _ => ()
                }
            }
            ast_map::NodeTraitItem(trait_item) => {
                visit::walk_trait_item(self, trait_item);
            }
            ast_map::NodeImplItem(impl_item) => {
                visit::walk_impl_item(self, impl_item);
            }
            ast_map::NodeForeignItem(foreign_item) => {
                visit::walk_foreign_item(self, &*foreign_item);
            }
            _ => ()
        }
        self.struct_has_extern_repr = had_extern_repr;
        self.inherited_pub_visibility = had_inherited_pub_visibility;
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for MarkSymbolVisitor<'a, 'tcx> {

    fn visit_variant_data(&mut self, def: &hir::VariantData, _: ast::Name,
                        _: &hir::Generics, _: ast::NodeId, _: codemap::Span) {
        let has_extern_repr = self.struct_has_extern_repr;
        let inherited_pub_visibility = self.inherited_pub_visibility;
        let live_fields = def.fields().iter().filter(|f| {
            has_extern_repr || inherited_pub_visibility || match f.node.kind {
                hir::NamedField(_, hir::Public) => true,
                _ => false
            }
        });
        self.live_symbols.extend(live_fields.map(|f| f.node.id));

        visit::walk_struct_def(self, def);
    }

    fn visit_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprMethodCall(..) => {
                self.lookup_and_handle_method(expr.id);
            }
            hir::ExprField(ref lhs, ref name) => {
                self.handle_field_access(&**lhs, name.node);
            }
            hir::ExprTupField(ref lhs, idx) => {
                self.handle_tup_field_access(&**lhs, idx.node);
            }
            _ => ()
        }

        visit::walk_expr(self, expr);
    }

    fn visit_arm(&mut self, arm: &hir::Arm) {
        if arm.pats.len() == 1 {
            let pat = &*arm.pats[0];
            let variants = pat_util::necessary_variants(&self.tcx.def_map.borrow(), pat);

            // Inside the body, ignore constructions of variants
            // necessary for the pattern to match. Those construction sites
            // can't be reached unless the variant is constructed elsewhere.
            let len = self.ignore_variant_stack.len();
            self.ignore_variant_stack.push_all(&*variants);
            visit::walk_arm(self, arm);
            self.ignore_variant_stack.truncate(len);
        } else {
            visit::walk_arm(self, arm);
        }
    }

    fn visit_pat(&mut self, pat: &hir::Pat) {
        let def_map = &self.tcx.def_map;
        match pat.node {
            hir::PatStruct(_, ref fields, _) => {
                self.handle_field_pattern_match(pat, fields);
            }
            _ if pat_util::pat_is_const(&def_map.borrow(), pat) => {
                // it might be the only use of a const
                self.lookup_and_handle_definition(&pat.id)
            }
            _ => ()
        }

        self.ignore_non_const_paths = true;
        visit::walk_pat(self, pat);
        self.ignore_non_const_paths = false;
    }

    fn visit_path(&mut self, path: &hir::Path, id: ast::NodeId) {
        self.lookup_and_handle_definition(&id);
        visit::walk_path(self, path);
    }

    fn visit_path_list_item(&mut self, path: &hir::Path, item: &hir::PathListItem) {
        self.lookup_and_handle_definition(&item.node.id());
        visit::walk_path_list_item(self, path, item);
    }

    fn visit_item(&mut self, _: &hir::Item) {
        // Do not recurse into items. These items will be added to the
        // worklist and recursed into manually if necessary.
    }
}

fn has_allow_dead_code_or_lang_attr(attrs: &[ast::Attribute]) -> bool {
    if attr::contains_name(attrs, "lang") {
        return true;
    }

    let dead_code = lint::builtin::DEAD_CODE.name_lower();
    for attr in lint::gather_attrs(attrs) {
        match attr {
            Ok((ref name, lint::Allow, _))
                if &name[..] == dead_code => return true,
            _ => (),
        }
    }
    false
}

// This visitor seeds items that
//   1) We want to explicitly consider as live:
//     * Item annotated with #[allow(dead_code)]
//         - This is done so that if we want to suppress warnings for a
//           group of dead functions, we only have to annotate the "root".
//           For example, if both `f` and `g` are dead and `f` calls `g`,
//           then annotating `f` with `#[allow(dead_code)]` will suppress
//           warning for both `f` and `g`.
//     * Item annotated with #[lang=".."]
//         - This is because lang items are always callable from elsewhere.
//   or
//   2) We are not sure to be live or not
//     * Implementation of a trait method
struct LifeSeeder {
    worklist: Vec<ast::NodeId>
}

impl<'v> Visitor<'v> for LifeSeeder {
    fn visit_item(&mut self, item: &hir::Item) {
        let allow_dead_code = has_allow_dead_code_or_lang_attr(&item.attrs);
        if allow_dead_code {
            self.worklist.push(item.id);
        }
        match item.node {
            hir::ItemEnum(ref enum_def, _) if allow_dead_code => {
                self.worklist.extend(enum_def.variants.iter()
                                                      .map(|variant| variant.node.data.id()));
            }
            hir::ItemTrait(_, _, _, ref trait_items) => {
                for trait_item in trait_items {
                    match trait_item.node {
                        hir::ConstTraitItem(_, Some(_)) |
                        hir::MethodTraitItem(_, Some(_)) => {
                            if has_allow_dead_code_or_lang_attr(&trait_item.attrs) {
                                self.worklist.push(trait_item.id);
                            }
                        }
                        _ => {}
                    }
                }
            }
            hir::ItemImpl(_, _, _, ref opt_trait, _, ref impl_items) => {
                for impl_item in impl_items {
                    match impl_item.node {
                        hir::ImplItem_::Const(..) |
                        hir::ImplItem_::Method(..) => {
                            if opt_trait.is_some() ||
                                    has_allow_dead_code_or_lang_attr(&impl_item.attrs) {
                                self.worklist.push(impl_item.id);
                            }
                        }
                        hir::ImplItem_::Type(_) => {}
                    }
                }
            }
            _ => ()
        }
        visit::walk_item(self, item);
    }
}

fn create_and_seed_worklist(tcx: &ty::ctxt,
                            exported_items: &privacy::ExportedItems,
                            reachable_symbols: &NodeSet,
                            krate: &hir::Crate) -> Vec<ast::NodeId> {
    let mut worklist = Vec::new();

    // Preferably, we would only need to seed the worklist with reachable
    // symbols. However, since the set of reachable symbols differs
    // depending on whether a crate is built as bin or lib, and we want
    // the warning to be consistent, we also seed the worklist with
    // exported symbols.
    for id in exported_items {
        worklist.push(*id);
    }
    for id in reachable_symbols {
        // Reachable variants can be dead, because we warn about
        // variants never constructed, not variants never used.
        if let Some(ast_map::NodeVariant(..)) = tcx.map.find(*id) {
            continue;
        }
        worklist.push(*id);
    }

    // Seed entry point
    match *tcx.sess.entry_fn.borrow() {
        Some((id, _)) => worklist.push(id),
        None => ()
    }

    // Seed implemented trait items
    let mut life_seeder = LifeSeeder {
        worklist: worklist
    };
    visit::walk_crate(&mut life_seeder, krate);

    return life_seeder.worklist;
}

fn find_live(tcx: &ty::ctxt,
             exported_items: &privacy::ExportedItems,
             reachable_symbols: &NodeSet,
             krate: &hir::Crate)
             -> Box<HashSet<ast::NodeId>> {
    let worklist = create_and_seed_worklist(tcx, exported_items,
                                            reachable_symbols, krate);
    let mut symbol_visitor = MarkSymbolVisitor::new(tcx, worklist);
    symbol_visitor.mark_live_symbols();
    symbol_visitor.live_symbols
}

fn get_struct_ctor_id(item: &hir::Item) -> Option<ast::NodeId> {
    match item.node {
        hir::ItemStruct(ref struct_def, _) if !struct_def.is_struct() => {
            Some(struct_def.id())
        }
        _ => None
    }
}

struct DeadVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    live_symbols: Box<HashSet<ast::NodeId>>,
}

impl<'a, 'tcx> DeadVisitor<'a, 'tcx> {
    fn should_warn_about_item(&mut self, item: &hir::Item) -> bool {
        let should_warn = match item.node {
            hir::ItemStatic(..)
            | hir::ItemConst(..)
            | hir::ItemFn(..)
            | hir::ItemEnum(..)
            | hir::ItemStruct(..) => true,
            _ => false
        };
        let ctor_id = get_struct_ctor_id(item);
        should_warn && !self.symbol_is_live(item.id, ctor_id)
    }

    fn should_warn_about_field(&mut self, node: &hir::StructField_) -> bool {
        let is_named = node.name().is_some();
        let field_type = self.tcx.node_id_to_type(node.id);
        let is_marker_field = match field_type.ty_to_def_id() {
            Some(def_id) => self.tcx.lang_items.items().any(|(_, item)| *item == Some(def_id)),
            _ => false
        };
        is_named
            && !self.symbol_is_live(node.id, None)
            && !is_marker_field
            && !has_allow_dead_code_or_lang_attr(&node.attrs)
    }

    fn should_warn_about_variant(&mut self, variant: &hir::Variant_) -> bool {
        !self.symbol_is_live(variant.data.id(), None)
            && !has_allow_dead_code_or_lang_attr(&variant.attrs)
    }

    // id := node id of an item's definition.
    // ctor_id := `Some` if the item is a struct_ctor (tuple struct),
    //            `None` otherwise.
    // If the item is a struct_ctor, then either its `id` or
    // `ctor_id` (unwrapped) is in the live_symbols set. More specifically,
    // DefMap maps the ExprPath of a struct_ctor to the node referred by
    // `ctor_id`. On the other hand, in a statement like
    // `type <ident> <generics> = <ty>;` where <ty> refers to a struct_ctor,
    // DefMap maps <ty> to `id` instead.
    fn symbol_is_live(&mut self,
                      id: ast::NodeId,
                      ctor_id: Option<ast::NodeId>)
                      -> bool {
        if self.live_symbols.contains(&id)
           || ctor_id.map_or(false,
                             |ctor| self.live_symbols.contains(&ctor)) {
            return true;
        }
        // If it's a type whose items are live, then it's live, too.
        // This is done to handle the case where, for example, the static
        // method of a private type is used, but the type itself is never
        // called directly.
        let impl_items = self.tcx.impl_items.borrow();
        match self.tcx.inherent_impls.borrow().get(&self.tcx.map.local_def_id(id)) {
            None => (),
            Some(impl_list) => {
                for impl_did in impl_list.iter() {
                    for item_did in impl_items.get(impl_did).unwrap().iter() {
                        if let Some(item_node_id) =
                                self.tcx.map.as_local_node_id(item_did.def_id()) {
                            if self.live_symbols.contains(&item_node_id) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn warn_dead_code(&mut self,
                      id: ast::NodeId,
                      span: codemap::Span,
                      name: ast::Name,
                      node_type: &str) {
        let name = name.as_str();
        if !name.starts_with("_") {
            self.tcx
                .sess
                .add_lint(lint::builtin::DEAD_CODE,
                          id,
                          span,
                          format!("{} is never used: `{}`", node_type, name));
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for DeadVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if self.should_warn_about_item(item) {
            self.warn_dead_code(
                item.id,
                item.span,
                item.name,
                item.node.descriptive_variant()
            );
        } else {
            // Only continue if we didn't warn
            visit::walk_item(self, item);
        }
    }

    fn visit_variant(&mut self, variant: &hir::Variant, g: &hir::Generics, id: ast::NodeId) {
        if self.should_warn_about_variant(&variant.node) {
            self.warn_dead_code(variant.node.data.id(), variant.span,
                                variant.node.name, "variant");
        } else {
            visit::walk_variant(self, variant, g, id);
        }
    }

    fn visit_foreign_item(&mut self, fi: &hir::ForeignItem) {
        if !self.symbol_is_live(fi.id, None) {
            self.warn_dead_code(fi.id, fi.span, fi.name, fi.node.descriptive_variant());
        }
        visit::walk_foreign_item(self, fi);
    }

    fn visit_struct_field(&mut self, field: &hir::StructField) {
        if self.should_warn_about_field(&field.node) {
            self.warn_dead_code(field.node.id, field.span,
                                field.node.name().unwrap(), "struct field");
        }

        visit::walk_struct_field(self, field);
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem) {
        match impl_item.node {
            hir::ImplItem_::Const(_, ref expr) => {
                if !self.symbol_is_live(impl_item.id, None) {
                    self.warn_dead_code(impl_item.id, impl_item.span,
                                        impl_item.name, "associated const");
                }
                visit::walk_expr(self, expr)
            }
            hir::ImplItem_::Method(_, ref body) => {
                if !self.symbol_is_live(impl_item.id, None) {
                    self.warn_dead_code(impl_item.id, impl_item.span,
                                        impl_item.name, "method");
                }
                visit::walk_block(self, body)
            }
            hir::ImplItem_::Type(..) => {}
        }
    }

    // Overwrite so that we don't warn the trait item itself.
    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem) {
        match trait_item.node {
            hir::ConstTraitItem(_, Some(ref expr)) => {
                visit::walk_expr(self, expr)
            }
            hir::MethodTraitItem(_, Some(ref body)) => {
                visit::walk_block(self, body)
            }
            hir::ConstTraitItem(_, None) |
            hir::MethodTraitItem(_, None) |
            hir::TypeTraitItem(..) => {}
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   exported_items: &privacy::ExportedItems,
                   reachable_symbols: &NodeSet) {
    let krate = tcx.map.krate();
    let live_symbols = find_live(tcx, exported_items,
                                 reachable_symbols, krate);
    let mut visitor = DeadVisitor { tcx: tcx, live_symbols: live_symbols };
    visit::walk_crate(&mut visitor, krate);
}
