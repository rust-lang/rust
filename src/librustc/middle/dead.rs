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

use dep_graph::DepNode;
use hir::map as ast_map;
use hir::{self, PatKind};
use hir::intravisit::{self, Visitor, NestedVisitorMap};
use hir::itemlikevisit::ItemLikeVisitor;

use middle::privacy;
use ty::{self, TyCtxt};
use hir::def::Def;
use hir::def_id::{DefId};
use lint;
use util::nodemap::FxHashSet;

use syntax::{ast, codemap};
use syntax::attr;
use syntax_pos;

// Any local node that may call something in its body block should be
// explored. For example, if it's a live NodeItem that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            node_id: ast::NodeId) -> bool {
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
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::Tables<'tcx>,
    live_symbols: Box<FxHashSet<ast::NodeId>>,
    struct_has_extern_repr: bool,
    ignore_non_const_paths: bool,
    inherited_pub_visibility: bool,
    ignore_variant_stack: Vec<DefId>,
}

impl<'a, 'tcx> MarkSymbolVisitor<'a, 'tcx> {
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

    fn handle_definition(&mut self, def: Def) {
        match def {
            Def::Const(_) | Def::AssociatedConst(..) => {
                self.check_def_id(def.def_id());
            }
            _ if self.ignore_non_const_paths => (),
            Def::PrimTy(..) | Def::SelfTy(..) => (),
            Def::Variant(variant_id) | Def::VariantCtor(variant_id, ..) => {
                if let Some(enum_id) = self.tcx.parent_def_id(variant_id) {
                    self.check_def_id(enum_id);
                }
                if !self.ignore_variant_stack.contains(&variant_id) {
                    self.check_def_id(variant_id);
                }
            }
            _ => {
                self.check_def_id(def.def_id());
            }
        }
    }

    fn lookup_and_handle_method(&mut self, id: ast::NodeId) {
        let method_call = ty::MethodCall::expr(id);
        let method = self.tables.method_map[&method_call];
        self.check_def_id(method.def_id);
    }

    fn handle_field_access(&mut self, lhs: &hir::Expr, name: ast::Name) {
        match self.tables.expr_ty_adjusted(lhs).sty {
            ty::TyAdt(def, _) => {
                self.insert_def_id(def.struct_variant().field_named(name).did);
            }
            _ => span_bug!(lhs.span, "named field access on non-ADT"),
        }
    }

    fn handle_tup_field_access(&mut self, lhs: &hir::Expr, idx: usize) {
        match self.tables.expr_ty_adjusted(lhs).sty {
            ty::TyAdt(def, _) => {
                self.insert_def_id(def.struct_variant().fields[idx].did);
            }
            ty::TyTuple(..) => {}
            _ => span_bug!(lhs.span, "numeric field access on non-ADT"),
        }
    }

    fn handle_field_pattern_match(&mut self, lhs: &hir::Pat, def: Def,
                                  pats: &[codemap::Spanned<hir::FieldPat>]) {
        let variant = match self.tables.node_id_to_type(lhs.id).sty {
            ty::TyAdt(adt, _) => adt.variant_of_def(def),
            _ => span_bug!(lhs.span, "non-ADT in struct pattern")
        };
        for pat in pats {
            if let PatKind::Wild = pat.node.pat.node {
                continue;
            }
            self.insert_def_id(variant.field_named(pat.node.name).did);
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = FxHashSet();
        while !self.worklist.is_empty() {
            let id = self.worklist.pop().unwrap();
            if scanned.contains(&id) {
                continue
            }
            scanned.insert(id);

            if let Some(ref node) = self.tcx.map.find(id) {
                self.live_symbols.insert(id);
                self.visit_node(node);
            }
        }
    }

    fn visit_node(&mut self, node: &ast_map::Node<'tcx>) {
        let had_extern_repr = self.struct_has_extern_repr;
        self.struct_has_extern_repr = false;
        let had_inherited_pub_visibility = self.inherited_pub_visibility;
        self.inherited_pub_visibility = false;
        match *node {
            ast_map::NodeItem(item) => {
                match item.node {
                    hir::ItemStruct(..) | hir::ItemUnion(..) => {
                        self.struct_has_extern_repr = item.attrs.iter().any(|attr| {
                            attr::find_repr_attrs(self.tcx.sess.diagnostic(), attr)
                                .contains(&attr::ReprExtern)
                        });

                        intravisit::walk_item(self, &item);
                    }
                    hir::ItemEnum(..) => {
                        self.inherited_pub_visibility = item.vis == hir::Public;
                        intravisit::walk_item(self, &item);
                    }
                    hir::ItemFn(..)
                    | hir::ItemTy(..)
                    | hir::ItemStatic(..)
                    | hir::ItemConst(..) => {
                        intravisit::walk_item(self, &item);
                    }
                    _ => ()
                }
            }
            ast_map::NodeTraitItem(trait_item) => {
                intravisit::walk_trait_item(self, trait_item);
            }
            ast_map::NodeImplItem(impl_item) => {
                intravisit::walk_impl_item(self, impl_item);
            }
            ast_map::NodeForeignItem(foreign_item) => {
                intravisit::walk_foreign_item(self, &foreign_item);
            }
            _ => ()
        }
        self.struct_has_extern_repr = had_extern_repr;
        self.inherited_pub_visibility = had_inherited_pub_visibility;
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MarkSymbolVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.map.body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_variant_data(&mut self, def: &'tcx hir::VariantData, _: ast::Name,
                        _: &hir::Generics, _: ast::NodeId, _: syntax_pos::Span) {
        let has_extern_repr = self.struct_has_extern_repr;
        let inherited_pub_visibility = self.inherited_pub_visibility;
        let live_fields = def.fields().iter().filter(|f| {
            has_extern_repr || inherited_pub_visibility || f.vis == hir::Public
        });
        self.live_symbols.extend(live_fields.map(|f| f.id));

        intravisit::walk_struct_def(self, def);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprPath(ref qpath @ hir::QPath::TypeRelative(..)) => {
                let def = self.tables.qpath_def(qpath, expr.id);
                self.handle_definition(def);
            }
            hir::ExprMethodCall(..) => {
                self.lookup_and_handle_method(expr.id);
            }
            hir::ExprField(ref lhs, ref name) => {
                self.handle_field_access(&lhs, name.node);
            }
            hir::ExprTupField(ref lhs, idx) => {
                self.handle_tup_field_access(&lhs, idx.node);
            }
            _ => ()
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm) {
        if arm.pats.len() == 1 {
            let variants = arm.pats[0].necessary_variants();

            // Inside the body, ignore constructions of variants
            // necessary for the pattern to match. Those construction sites
            // can't be reached unless the variant is constructed elsewhere.
            let len = self.ignore_variant_stack.len();
            self.ignore_variant_stack.extend_from_slice(&variants);
            intravisit::walk_arm(self, arm);
            self.ignore_variant_stack.truncate(len);
        } else {
            intravisit::walk_arm(self, arm);
        }
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat) {
        match pat.node {
            PatKind::Struct(hir::QPath::Resolved(_, ref path), ref fields, _) => {
                self.handle_field_pattern_match(pat, path.def, fields);
            }
            PatKind::Path(ref qpath @ hir::QPath::TypeRelative(..)) => {
                let def = self.tables.qpath_def(qpath, pat.id);
                self.handle_definition(def);
            }
            _ => ()
        }

        self.ignore_non_const_paths = true;
        intravisit::walk_pat(self, pat);
        self.ignore_non_const_paths = false;
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, _: ast::NodeId) {
        self.handle_definition(path.def);
        intravisit::walk_path(self, path);
    }
}

fn has_allow_dead_code_or_lang_attr(attrs: &[ast::Attribute]) -> bool {
    if attr::contains_name(attrs, "lang") {
        return true;
    }

    let dead_code = lint::builtin::DEAD_CODE.name_lower();
    for attr in lint::gather_attrs(attrs) {
        match attr {
            Ok((name, lint::Allow, _)) if name == &*dead_code => return true,
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
struct LifeSeeder<'k> {
    worklist: Vec<ast::NodeId>,
    krate: &'k hir::Crate,
}

impl<'v, 'k> ItemLikeVisitor<'v> for LifeSeeder<'k> {
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
            hir::ItemTrait(.., ref trait_item_refs) => {
                for trait_item_ref in trait_item_refs {
                    let trait_item = self.krate.trait_item(trait_item_ref.id);
                    match trait_item.node {
                        hir::TraitItemKind::Const(_, Some(_)) |
                        hir::TraitItemKind::Method(_, hir::TraitMethod::Provided(_)) => {
                            if has_allow_dead_code_or_lang_attr(&trait_item.attrs) {
                                self.worklist.push(trait_item.id);
                            }
                        }
                        _ => {}
                    }
                }
            }
            hir::ItemImpl(.., ref opt_trait, _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    let impl_item = self.krate.impl_item(impl_item_ref.id);
                    if opt_trait.is_some() ||
                            has_allow_dead_code_or_lang_attr(&impl_item.attrs) {
                        self.worklist.push(impl_item_ref.id.node_id);
                    }
                }
            }
            _ => ()
        }
    }

    fn visit_trait_item(&mut self, _item: &hir::TraitItem) {
        // ignore: we are handling this in `visit_item` above
    }

    fn visit_impl_item(&mut self, _item: &hir::ImplItem) {
        // ignore: we are handling this in `visit_item` above
    }
}

fn create_and_seed_worklist<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      access_levels: &privacy::AccessLevels,
                                      krate: &hir::Crate)
                                      -> Vec<ast::NodeId> {
    let mut worklist = Vec::new();
    for (id, _) in &access_levels.map {
        worklist.push(*id);
    }

    // Seed entry point
    if let Some((id, _)) = *tcx.sess.entry_fn.borrow() {
        worklist.push(id);
    }

    // Seed implemented trait items
    let mut life_seeder = LifeSeeder {
        worklist: worklist,
        krate: krate,
    };
    krate.visit_all_item_likes(&mut life_seeder);

    return life_seeder.worklist;
}

fn find_live<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       access_levels: &privacy::AccessLevels,
                       krate: &hir::Crate)
                       -> Box<FxHashSet<ast::NodeId>> {
    let worklist = create_and_seed_worklist(tcx, access_levels, krate);
    let mut symbol_visitor = MarkSymbolVisitor {
        worklist: worklist,
        tcx: tcx,
        tables: &ty::Tables::empty(),
        live_symbols: box FxHashSet(),
        struct_has_extern_repr: false,
        ignore_non_const_paths: false,
        inherited_pub_visibility: false,
        ignore_variant_stack: vec![],
    };
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
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    live_symbols: Box<FxHashSet<ast::NodeId>>,
}

impl<'a, 'tcx> DeadVisitor<'a, 'tcx> {
    fn should_warn_about_item(&mut self, item: &hir::Item) -> bool {
        let should_warn = match item.node {
            hir::ItemStatic(..)
            | hir::ItemConst(..)
            | hir::ItemFn(..)
            | hir::ItemTy(..)
            | hir::ItemEnum(..)
            | hir::ItemStruct(..)
            | hir::ItemUnion(..) => true,
            _ => false
        };
        let ctor_id = get_struct_ctor_id(item);
        should_warn && !self.symbol_is_live(item.id, ctor_id)
    }

    fn should_warn_about_field(&mut self, field: &hir::StructField) -> bool {
        let field_type = self.tcx.item_type(self.tcx.map.local_def_id(field.id));
        let is_marker_field = match field_type.ty_to_def_id() {
            Some(def_id) => self.tcx.lang_items.items().iter().any(|item| *item == Some(def_id)),
            _ => false
        };
        !field.is_positional()
            && !self.symbol_is_live(field.id, None)
            && !is_marker_field
            && !has_allow_dead_code_or_lang_attr(&field.attrs)
    }

    fn should_warn_about_variant(&mut self, variant: &hir::Variant_) -> bool {
        !self.symbol_is_live(variant.data.id(), None)
            && !has_allow_dead_code_or_lang_attr(&variant.attrs)
    }

    fn should_warn_about_foreign_item(&mut self, fi: &hir::ForeignItem) -> bool {
        !self.symbol_is_live(fi.id, None)
            && !has_allow_dead_code_or_lang_attr(&fi.attrs)
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
        if let Some(impl_list) =
                self.tcx.inherent_impls.borrow().get(&self.tcx.map.local_def_id(id)) {
            for &impl_did in impl_list.iter() {
                for &item_did in &self.tcx.associated_item_def_ids(impl_did)[..] {
                    if let Some(item_node_id) = self.tcx.map.as_local_node_id(item_did) {
                        if self.live_symbols.contains(&item_node_id) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn warn_dead_code(&mut self,
                      id: ast::NodeId,
                      span: syntax_pos::Span,
                      name: ast::Name,
                      node_type: &str) {
        if !name.as_str().starts_with("_") {
            self.tcx
                .sess
                .add_lint(lint::builtin::DEAD_CODE,
                          id,
                          span,
                          format!("{} is never used: `{}`", node_type, name));
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for DeadVisitor<'a, 'tcx> {
    /// Walk nested items in place so that we don't report dead-code
    /// on inner functions when the outer function is already getting
    /// an error. We could do this also by checking the parents, but
    /// this is how the code is setup and it seems harmless enough.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.map)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        if self.should_warn_about_item(item) {
            self.warn_dead_code(
                item.id,
                item.span,
                item.name,
                item.node.descriptive_variant()
            );
        } else {
            // Only continue if we didn't warn
            intravisit::walk_item(self, item);
        }
    }

    fn visit_variant(&mut self,
                     variant: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     id: ast::NodeId) {
        if self.should_warn_about_variant(&variant.node) {
            self.warn_dead_code(variant.node.data.id(), variant.span,
                                variant.node.name, "variant");
        } else {
            intravisit::walk_variant(self, variant, g, id);
        }
    }

    fn visit_foreign_item(&mut self, fi: &'tcx hir::ForeignItem) {
        if self.should_warn_about_foreign_item(fi) {
            self.warn_dead_code(fi.id, fi.span, fi.name, fi.node.descriptive_variant());
        }
        intravisit::walk_foreign_item(self, fi);
    }

    fn visit_struct_field(&mut self, field: &'tcx hir::StructField) {
        if self.should_warn_about_field(&field) {
            self.warn_dead_code(field.id, field.span,
                                field.name, "field");
        }

        intravisit::walk_struct_field(self, field);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        match impl_item.node {
            hir::ImplItemKind::Const(_, body_id) => {
                if !self.symbol_is_live(impl_item.id, None) {
                    self.warn_dead_code(impl_item.id, impl_item.span,
                                        impl_item.name, "associated const");
                }
                self.visit_nested_body(body_id)
            }
            hir::ImplItemKind::Method(_, body_id) => {
                if !self.symbol_is_live(impl_item.id, None) {
                    self.warn_dead_code(impl_item.id, impl_item.span,
                                        impl_item.name, "method");
                }
                self.visit_nested_body(body_id)
            }
            hir::ImplItemKind::Type(..) => {}
        }
    }

    // Overwrite so that we don't warn the trait item itself.
    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        match trait_item.node {
            hir::TraitItemKind::Const(_, Some(body_id)) |
            hir::TraitItemKind::Method(_, hir::TraitMethod::Provided(body_id)) => {
                self.visit_nested_body(body_id)
            }
            hir::TraitItemKind::Const(_, None) |
            hir::TraitItemKind::Method(_, hir::TraitMethod::Required(_)) |
            hir::TraitItemKind::Type(..) => {}
        }
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             access_levels: &privacy::AccessLevels) {
    let _task = tcx.dep_graph.in_task(DepNode::DeadCheck);
    let krate = tcx.map.krate();
    let live_symbols = find_live(tcx, access_levels, krate);
    let mut visitor = DeadVisitor { tcx: tcx, live_symbols: live_symbols };
    intravisit::walk_crate(&mut visitor, krate);
}
