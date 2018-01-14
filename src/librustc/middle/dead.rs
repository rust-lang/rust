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

use hir::map as hir_map;
use hir::{self, Item_, PatKind};
use hir::intravisit::{self, Visitor, NestedVisitorMap};
use hir::itemlikevisit::ItemLikeVisitor;

use hir::def::Def;
use hir::def_id::{DefId, LOCAL_CRATE};
use lint;
use middle::privacy;
use ty::{self, TyCtxt};
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
    match tcx.hir.find(node_id) {
        Some(hir_map::NodeItem(..)) |
        Some(hir_map::NodeImplItem(..)) |
        Some(hir_map::NodeForeignItem(..)) |
        Some(hir_map::NodeTraitItem(..)) =>
            true,
        _ =>
            false
    }
}

struct MarkSymbolVisitor<'a, 'tcx: 'a> {
    worklist: Vec<ast::NodeId>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    live_symbols: Box<FxHashSet<ast::NodeId>>,
    repr_has_repr_c: bool,
    in_pat: bool,
    inherited_pub_visibility: bool,
    ignore_variant_stack: Vec<DefId>,
}

impl<'a, 'tcx> MarkSymbolVisitor<'a, 'tcx> {
    fn check_def_id(&mut self, def_id: DefId) {
        if let Some(node_id) = self.tcx.hir.as_local_node_id(def_id) {
            if should_explore(self.tcx, node_id) {
                self.worklist.push(node_id);
            }
            self.live_symbols.insert(node_id);
        }
    }

    fn insert_def_id(&mut self, def_id: DefId) {
        if let Some(node_id) = self.tcx.hir.as_local_node_id(def_id) {
            debug_assert!(!should_explore(self.tcx, node_id));
            self.live_symbols.insert(node_id);
        }
    }

    fn handle_definition(&mut self, def: Def) {
        match def {
            Def::Const(_) | Def::AssociatedConst(..) | Def::TyAlias(_) => {
                self.check_def_id(def.def_id());
            }
            _ if self.in_pat => (),
            Def::PrimTy(..) | Def::SelfTy(..) |
            Def::Local(..) | Def::Upvar(..) => {}
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

    fn lookup_and_handle_method(&mut self, id: hir::HirId) {
        self.check_def_id(self.tables.type_dependent_defs()[id].def_id());
    }

    fn handle_field_access(&mut self, lhs: &hir::Expr, name: ast::Name) {
        match self.tables.expr_ty_adjusted(lhs).sty {
            ty::TyAdt(def, _) => {
                self.insert_def_id(def.non_enum_variant().field_named(name).did);
            }
            _ => span_bug!(lhs.span, "named field access on non-ADT"),
        }
    }

    fn handle_tup_field_access(&mut self, lhs: &hir::Expr, idx: usize) {
        match self.tables.expr_ty_adjusted(lhs).sty {
            ty::TyAdt(def, _) => {
                self.insert_def_id(def.non_enum_variant().fields[idx].did);
            }
            ty::TyTuple(..) => {}
            _ => span_bug!(lhs.span, "numeric field access on non-ADT"),
        }
    }

    fn handle_field_pattern_match(&mut self, lhs: &hir::Pat, def: Def,
                                  pats: &[codemap::Spanned<hir::FieldPat>]) {
        let variant = match self.tables.node_id_to_type(lhs.hir_id).sty {
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

            if let Some(ref node) = self.tcx.hir.find(id) {
                self.live_symbols.insert(id);
                self.visit_node(node);
            }
        }
    }

    fn visit_node(&mut self, node: &hir_map::Node<'tcx>) {
        let had_repr_c = self.repr_has_repr_c;
        self.repr_has_repr_c = false;
        let had_inherited_pub_visibility = self.inherited_pub_visibility;
        self.inherited_pub_visibility = false;
        match *node {
            hir_map::NodeItem(item) => {
                match item.node {
                    hir::ItemStruct(..) | hir::ItemUnion(..) => {
                        let def_id = self.tcx.hir.local_def_id(item.id);
                        let def = self.tcx.adt_def(def_id);
                        self.repr_has_repr_c = def.repr.c();

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
            hir_map::NodeTraitItem(trait_item) => {
                intravisit::walk_trait_item(self, trait_item);
            }
            hir_map::NodeImplItem(impl_item) => {
                intravisit::walk_impl_item(self, impl_item);
            }
            hir_map::NodeForeignItem(foreign_item) => {
                intravisit::walk_foreign_item(self, &foreign_item);
            }
            _ => ()
        }
        self.repr_has_repr_c = had_repr_c;
        self.inherited_pub_visibility = had_inherited_pub_visibility;
    }

    fn mark_as_used_if_union(&mut self, did: DefId, fields: &hir::HirVec<hir::Field>) {
        if let Some(node_id) = self.tcx.hir.as_local_node_id(did) {
            if let Some(hir_map::NodeItem(item)) = self.tcx.hir.find(node_id) {
                if let Item_::ItemUnion(ref variant, _) = item.node {
                    if variant.fields().len() > 1 {
                        for field in variant.fields() {
                            if fields.iter().find(|x| x.name.node == field.name).is_some() {
                                self.live_symbols.insert(field.id);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MarkSymbolVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.hir.body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_variant_data(&mut self, def: &'tcx hir::VariantData, _: ast::Name,
                        _: &hir::Generics, _: ast::NodeId, _: syntax_pos::Span) {
        let has_repr_c = self.repr_has_repr_c;
        let inherited_pub_visibility = self.inherited_pub_visibility;
        let live_fields = def.fields().iter().filter(|f| {
            has_repr_c || inherited_pub_visibility || f.vis == hir::Public
        });
        self.live_symbols.extend(live_fields.map(|f| f.id));

        intravisit::walk_struct_def(self, def);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprPath(ref qpath @ hir::QPath::TypeRelative(..)) => {
                let def = self.tables.qpath_def(qpath, expr.hir_id);
                self.handle_definition(def);
            }
            hir::ExprMethodCall(..) => {
                self.lookup_and_handle_method(expr.hir_id);
            }
            hir::ExprField(ref lhs, ref name) => {
                self.handle_field_access(&lhs, name.node);
            }
            hir::ExprTupField(ref lhs, idx) => {
                self.handle_tup_field_access(&lhs, idx.node);
            }
            hir::ExprStruct(_, ref fields, _) => {
                if let ty::TypeVariants::TyAdt(ref def, _) = self.tables.expr_ty(expr).sty {
                    if def.is_union() {
                        self.mark_as_used_if_union(def.did, fields);
                    }
                }
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
                let def = self.tables.qpath_def(qpath, pat.hir_id);
                self.handle_definition(def);
            }
            _ => ()
        }

        self.in_pat = true;
        intravisit::walk_pat(self, pat);
        self.in_pat = false;
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, _: ast::NodeId) {
        self.handle_definition(path.def);
        intravisit::walk_path(self, path);
    }
}

fn has_allow_dead_code_or_lang_attr(tcx: TyCtxt,
                                    id: ast::NodeId,
                                    attrs: &[ast::Attribute]) -> bool {
    if attr::contains_name(attrs, "lang") {
        return true;
    }

    // #[used] also keeps the item alive forcefully,
    // e.g. for placing it in a specific section.
    if attr::contains_name(attrs, "used") {
        return true;
    }

    // Don't lint about global allocators
    if attr::contains_name(attrs, "global_allocator") {
        return true;
    }

    tcx.lint_level_at_node(lint::builtin::DEAD_CODE, id).0 == lint::Allow
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
struct LifeSeeder<'k, 'tcx: 'k> {
    worklist: Vec<ast::NodeId>,
    krate: &'k hir::Crate,
    tcx: TyCtxt<'k, 'tcx, 'tcx>,
}

impl<'v, 'k, 'tcx> ItemLikeVisitor<'v> for LifeSeeder<'k, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        let allow_dead_code = has_allow_dead_code_or_lang_attr(self.tcx,
                                                               item.id,
                                                               &item.attrs);
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
                            if has_allow_dead_code_or_lang_attr(self.tcx,
                                                                trait_item.id,
                                                                &trait_item.attrs) {
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
                            has_allow_dead_code_or_lang_attr(self.tcx,
                                                             impl_item.id,
                                                             &impl_item.attrs) {
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
        worklist,
        krate,
        tcx,
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
        worklist,
        tcx,
        tables: &ty::TypeckTables::empty(None),
        live_symbols: box FxHashSet(),
        repr_has_repr_c: false,
        in_pat: false,
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
        let field_type = self.tcx.type_of(self.tcx.hir.local_def_id(field.id));
        let is_marker_field = match field_type.ty_to_def_id() {
            Some(def_id) => self.tcx.lang_items().items().iter().any(|item| *item == Some(def_id)),
            _ => false
        };
        !field.is_positional()
            && !self.symbol_is_live(field.id, None)
            && !is_marker_field
            && !has_allow_dead_code_or_lang_attr(self.tcx, field.id, &field.attrs)
    }

    fn should_warn_about_variant(&mut self, variant: &hir::Variant_) -> bool {
        !self.symbol_is_live(variant.data.id(), None)
            && !has_allow_dead_code_or_lang_attr(self.tcx,
                                                 variant.data.id(),
                                                 &variant.attrs)
    }

    fn should_warn_about_foreign_item(&mut self, fi: &hir::ForeignItem) -> bool {
        !self.symbol_is_live(fi.id, None)
            && !has_allow_dead_code_or_lang_attr(self.tcx, fi.id, &fi.attrs)
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
        let def_id = self.tcx.hir.local_def_id(id);
        let inherent_impls = self.tcx.inherent_impls(def_id);
        for &impl_did in inherent_impls.iter() {
            for &item_did in &self.tcx.associated_item_def_ids(impl_did)[..] {
                if let Some(item_node_id) = self.tcx.hir.as_local_node_id(item_did) {
                    if self.live_symbols.contains(&item_node_id) {
                        return true;
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
                      node_type: &str,
                      participle: &str) {
        if !name.as_str().starts_with("_") {
            self.tcx
                .lint_node(lint::builtin::DEAD_CODE,
                           id,
                           span,
                           &format!("{} is never {}: `{}`",
                                    node_type, participle, name));
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for DeadVisitor<'a, 'tcx> {
    /// Walk nested items in place so that we don't report dead-code
    /// on inner functions when the outer function is already getting
    /// an error. We could do this also by checking the parents, but
    /// this is how the code is setup and it seems harmless enough.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        if self.should_warn_about_item(item) {
            // For items that have a definition with a signature followed by a
            // block, point only at the signature.
            let span = match item.node {
                hir::ItemFn(..) |
                hir::ItemMod(..) |
                hir::ItemEnum(..) |
                hir::ItemStruct(..) |
                hir::ItemUnion(..) |
                hir::ItemTrait(..) |
                hir::ItemImpl(..) => self.tcx.sess.codemap().def_span(item.span),
                _ => item.span,
            };
            self.warn_dead_code(
                item.id,
                span,
                item.name,
                item.node.descriptive_variant(),
                "used",
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
            self.warn_dead_code(variant.node.data.id(), variant.span, variant.node.name,
                                "variant", "constructed");
        } else {
            intravisit::walk_variant(self, variant, g, id);
        }
    }

    fn visit_foreign_item(&mut self, fi: &'tcx hir::ForeignItem) {
        if self.should_warn_about_foreign_item(fi) {
            self.warn_dead_code(fi.id, fi.span, fi.name,
                                fi.node.descriptive_variant(), "used");
        }
        intravisit::walk_foreign_item(self, fi);
    }

    fn visit_struct_field(&mut self, field: &'tcx hir::StructField) {
        if self.should_warn_about_field(&field) {
            self.warn_dead_code(field.id, field.span, field.name, "field", "used");
        }
        intravisit::walk_struct_field(self, field);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        match impl_item.node {
            hir::ImplItemKind::Const(_, body_id) => {
                if !self.symbol_is_live(impl_item.id, None) {
                    self.warn_dead_code(impl_item.id,
                                        impl_item.span,
                                        impl_item.name,
                                        "associated const",
                                        "used");
                }
                self.visit_nested_body(body_id)
            }
            hir::ImplItemKind::Method(_, body_id) => {
                if !self.symbol_is_live(impl_item.id, None) {
                    let span = self.tcx.sess.codemap().def_span(impl_item.span);
                    self.warn_dead_code(impl_item.id, span, impl_item.name, "method", "used");
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

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);
    let krate = tcx.hir.krate();
    let live_symbols = find_live(tcx, access_levels, krate);
    let mut visitor = DeadVisitor {
        tcx,
        live_symbols,
    };
    intravisit::walk_crate(&mut visitor, krate);
}
