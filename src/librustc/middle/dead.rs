// This implements the dead-code warning pass. It follows middle::reachable
// closely. The idea is that all reachable symbols are live, codes called
// from live codes are live, and everything else is dead.

use crate::hir::Node;
use crate::hir::{self, PatKind, TyKind};
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::hir::itemlikevisit::ItemLikeVisitor;

use crate::hir::def::{CtorOf, Res, DefKind};
use crate::hir::CodegenFnAttrFlags;
use crate::hir::def_id::{DefId, LOCAL_CRATE};
use crate::lint;
use crate::middle::privacy;
use crate::ty::{self, DefIdTree, TyCtxt};
use crate::util::nodemap::FxHashSet;

use rustc_data_structures::fx::FxHashMap;

use syntax::{ast, source_map};
use syntax::attr;
use syntax::symbol::sym;
use syntax_pos;

// Any local node that may call something in its body block should be
// explored. For example, if it's a live Node::Item that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    match tcx.hir().find(hir_id) {
        Some(Node::Item(..)) |
        Some(Node::ImplItem(..)) |
        Some(Node::ForeignItem(..)) |
        Some(Node::TraitItem(..)) =>
            true,
        _ =>
            false
    }
}

struct MarkSymbolVisitor<'a, 'tcx> {
    worklist: Vec<hir::HirId>,
    tcx: TyCtxt<'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    live_symbols: FxHashSet<hir::HirId>,
    repr_has_repr_c: bool,
    in_pat: bool,
    inherited_pub_visibility: bool,
    ignore_variant_stack: Vec<DefId>,
    // maps from tuple struct constructors to tuple struct items
    struct_constructors: FxHashMap<hir::HirId, hir::HirId>,
}

impl<'a, 'tcx> MarkSymbolVisitor<'a, 'tcx> {
    fn check_def_id(&mut self, def_id: DefId) {
        if let Some(hir_id) = self.tcx.hir().as_local_hir_id(def_id) {
            if should_explore(self.tcx, hir_id) || self.struct_constructors.contains_key(&hir_id) {
                self.worklist.push(hir_id);
            }
            self.live_symbols.insert(hir_id);
        }
    }

    fn insert_def_id(&mut self, def_id: DefId) {
        if let Some(hir_id) = self.tcx.hir().as_local_hir_id(def_id) {
            debug_assert!(!should_explore(self.tcx, hir_id));
            self.live_symbols.insert(hir_id);
        }
    }

    fn handle_res(&mut self, res: Res) {
        match res {
            Res::Def(DefKind::Const, _)
            | Res::Def(DefKind::AssocConst, _)
            | Res::Def(DefKind::TyAlias, _) => {
                self.check_def_id(res.def_id());
            }
            _ if self.in_pat => {},
            Res::PrimTy(..) | Res::SelfTy(..) | Res::SelfCtor(..) |
            Res::Local(..) => {}
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), ctor_def_id) => {
                let variant_id = self.tcx.parent(ctor_def_id).unwrap();
                let enum_id = self.tcx.parent(variant_id).unwrap();
                self.check_def_id(enum_id);
                if !self.ignore_variant_stack.contains(&ctor_def_id) {
                    self.check_def_id(variant_id);
                }
            }
            Res::Def(DefKind::Variant, variant_id) => {
                let enum_id = self.tcx.parent(variant_id).unwrap();
                self.check_def_id(enum_id);
                if !self.ignore_variant_stack.contains(&variant_id) {
                    self.check_def_id(variant_id);
                }
            }
            Res::ToolMod | Res::NonMacroAttr(..) | Res::Err => {}
            _ => {
                self.check_def_id(res.def_id());
            }
        }
    }

    fn lookup_and_handle_method(&mut self, id: hir::HirId) {
        if let Some(def_id) = self.tables.type_dependent_def_id(id) {
            self.check_def_id(def_id);
        } else {
            bug!("no type-dependent def for method");
        }
    }

    fn handle_field_access(&mut self, lhs: &hir::Expr, hir_id: hir::HirId) {
        match self.tables.expr_ty_adjusted(lhs).sty {
            ty::Adt(def, _) => {
                let index = self.tcx.field_index(hir_id, self.tables);
                self.insert_def_id(def.non_enum_variant().fields[index].did);
            }
            ty::Tuple(..) => {}
            _ => span_bug!(lhs.span, "named field access on non-ADT"),
        }
    }

    fn handle_field_pattern_match(&mut self, lhs: &hir::Pat, res: Res,
                                  pats: &[source_map::Spanned<hir::FieldPat>]) {
        let variant = match self.tables.node_type(lhs.hir_id).sty {
            ty::Adt(adt, _) => adt.variant_of_res(res),
            _ => span_bug!(lhs.span, "non-ADT in struct pattern")
        };
        for pat in pats {
            if let PatKind::Wild = pat.node.pat.node {
                continue;
            }
            let index = self.tcx.field_index(pat.node.hir_id, self.tables);
            self.insert_def_id(variant.fields[index].did);
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = FxHashSet::default();
        while let Some(id) = self.worklist.pop() {
            if !scanned.insert(id) {
                continue
            }

            // in the case of tuple struct constructors we want to check the item, not the generated
            // tuple struct constructor function
            let id = self.struct_constructors.get(&id).cloned().unwrap_or(id);

            if let Some(node) = self.tcx.hir().find(id) {
                self.live_symbols.insert(id);
                self.visit_node(node);
            }
        }
    }

    fn visit_node(&mut self, node: Node<'tcx>) {
        let had_repr_c = self.repr_has_repr_c;
        self.repr_has_repr_c = false;
        let had_inherited_pub_visibility = self.inherited_pub_visibility;
        self.inherited_pub_visibility = false;
        match node {
            Node::Item(item) => {
                match item.node {
                    hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) => {
                        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                        let def = self.tcx.adt_def(def_id);
                        self.repr_has_repr_c = def.repr.c();

                        intravisit::walk_item(self, &item);
                    }
                    hir::ItemKind::Enum(..) => {
                        self.inherited_pub_visibility = item.vis.node.is_pub();

                        intravisit::walk_item(self, &item);
                    }
                    hir::ItemKind::ForeignMod(..) => {}
                    _ => {
                        intravisit::walk_item(self, &item);
                    }
                }
            }
            Node::TraitItem(trait_item) => {
                intravisit::walk_trait_item(self, trait_item);
            }
            Node::ImplItem(impl_item) => {
                intravisit::walk_impl_item(self, impl_item);
            }
            Node::ForeignItem(foreign_item) => {
                intravisit::walk_foreign_item(self, &foreign_item);
            }
            _ => {}
        }
        self.repr_has_repr_c = had_repr_c;
        self.inherited_pub_visibility = had_inherited_pub_visibility;
    }

    fn mark_as_used_if_union(&mut self, adt: &ty::AdtDef, fields: &hir::HirVec<hir::Field>) {
        if adt.is_union() && adt.non_enum_variant().fields.len() > 1 && adt.did.is_local() {
            for field in fields {
                let index = self.tcx.field_index(field.hir_id, self.tables);
                self.insert_def_id(adt.non_enum_variant().fields[index].did);
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
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_variant_data(&mut self, def: &'tcx hir::VariantData, _: ast::Name,
                          _: &hir::Generics, _: hir::HirId, _: syntax_pos::Span) {
        let has_repr_c = self.repr_has_repr_c;
        let inherited_pub_visibility = self.inherited_pub_visibility;
        let live_fields = def.fields().iter().filter(|f| {
            has_repr_c || inherited_pub_visibility || f.vis.node.is_pub()
        });
        self.live_symbols.extend(live_fields.map(|f| f.hir_id));

        intravisit::walk_struct_def(self, def);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprKind::Path(ref qpath @ hir::QPath::TypeRelative(..)) => {
                let res = self.tables.qpath_res(qpath, expr.hir_id);
                self.handle_res(res);
            }
            hir::ExprKind::MethodCall(..) => {
                self.lookup_and_handle_method(expr.hir_id);
            }
            hir::ExprKind::Field(ref lhs, ..) => {
                self.handle_field_access(&lhs, expr.hir_id);
            }
            hir::ExprKind::Struct(_, ref fields, _) => {
                if let ty::Adt(ref adt, _) = self.tables.expr_ty(expr).sty {
                    self.mark_as_used_if_union(adt, fields);
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
                self.handle_field_pattern_match(pat, path.res, fields);
            }
            PatKind::Path(ref qpath @ hir::QPath::TypeRelative(..)) => {
                let res = self.tables.qpath_res(qpath, pat.hir_id);
                self.handle_res(res);
            }
            _ => ()
        }

        self.in_pat = true;
        intravisit::walk_pat(self, pat);
        self.in_pat = false;
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, _: hir::HirId) {
        self.handle_res(path.res);
        intravisit::walk_path(self, path);
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        match ty.node {
            TyKind::Def(item_id, _) => {
                let item = self.tcx.hir().expect_item(item_id.id);
                intravisit::walk_item(self, item);
            }
            _ => ()
        }
        intravisit::walk_ty(self, ty);
    }
}

fn has_allow_dead_code_or_lang_attr(
    tcx: TyCtxt<'_>,
    id: hir::HirId,
    attrs: &[ast::Attribute],
) -> bool {
    if attr::contains_name(attrs, sym::lang) {
        return true;
    }

    // Stable attribute for #[lang = "panic_impl"]
    if attr::contains_name(attrs, sym::panic_handler) {
        return true;
    }

    // (To be) stable attribute for #[lang = "oom"]
    if attr::contains_name(attrs, sym::alloc_error_handler) {
        return true;
    }

    // Don't lint about global allocators
    if attr::contains_name(attrs, sym::global_allocator) {
        return true;
    }

    let def_id = tcx.hir().local_def_id_from_hir_id(id);
    let cg_attrs = tcx.codegen_fn_attrs(def_id);

    // #[used], #[no_mangle], #[export_name], etc also keeps the item alive
    // forcefully, e.g., for placing it in a specific section.
    if cg_attrs.contains_extern_indicator() ||
        cg_attrs.flags.contains(CodegenFnAttrFlags::USED) {
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
struct LifeSeeder<'k, 'tcx> {
    worklist: Vec<hir::HirId>,
    krate: &'k hir::Crate,
    tcx: TyCtxt<'tcx>,
    // see `MarkSymbolVisitor::struct_constructors`
    struct_constructors: FxHashMap<hir::HirId, hir::HirId>,
}

impl<'v, 'k, 'tcx> ItemLikeVisitor<'v> for LifeSeeder<'k, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        let allow_dead_code = has_allow_dead_code_or_lang_attr(self.tcx,
                                                               item.hir_id,
                                                               &item.attrs);
        if allow_dead_code {
            self.worklist.push(item.hir_id);
        }
        match item.node {
            hir::ItemKind::Enum(ref enum_def, _) => {
                if allow_dead_code {
                    self.worklist.extend(enum_def.variants.iter().map(|variant| variant.node.id));
                }

                for variant in &enum_def.variants {
                    if let Some(ctor_hir_id) = variant.node.data.ctor_hir_id() {
                        self.struct_constructors.insert(ctor_hir_id, variant.node.id);
                    }
                }
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                for trait_item_ref in trait_item_refs {
                    let trait_item = self.krate.trait_item(trait_item_ref.id);
                    match trait_item.node {
                        hir::TraitItemKind::Const(_, Some(_)) |
                        hir::TraitItemKind::Method(_, hir::TraitMethod::Provided(_)) => {
                            if has_allow_dead_code_or_lang_attr(self.tcx,
                                                                trait_item.hir_id,
                                                                &trait_item.attrs) {
                                self.worklist.push(trait_item.hir_id);
                            }
                        }
                        _ => {}
                    }
                }
            }
            hir::ItemKind::Impl(.., ref opt_trait, _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    let impl_item = self.krate.impl_item(impl_item_ref.id);
                    if opt_trait.is_some() ||
                            has_allow_dead_code_or_lang_attr(self.tcx,
                                                             impl_item.hir_id,
                                                             &impl_item.attrs) {
                        self.worklist.push(impl_item_ref.id.hir_id);
                    }
                }
            }
            hir::ItemKind::Struct(ref variant_data, _) => {
                if let Some(ctor_hir_id) = variant_data.ctor_hir_id() {
                    self.struct_constructors.insert(ctor_hir_id, item.hir_id);
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

fn create_and_seed_worklist<'tcx>(
    tcx: TyCtxt<'tcx>,
    access_levels: &privacy::AccessLevels,
    krate: &hir::Crate,
) -> (Vec<hir::HirId>, FxHashMap<hir::HirId, hir::HirId>) {
    let worklist = access_levels.map.iter().filter_map(|(&id, level)| {
        if level >= &privacy::AccessLevel::Reachable {
            Some(id)
        } else {
            None
        }
    }).chain(
        // Seed entry point
        tcx.entry_fn(LOCAL_CRATE).map(|(def_id, _)| tcx.hir().as_local_hir_id(def_id).unwrap())
    ).collect::<Vec<_>>();

    // Seed implemented trait items
    let mut life_seeder = LifeSeeder {
        worklist,
        krate,
        tcx,
        struct_constructors: Default::default(),
    };
    krate.visit_all_item_likes(&mut life_seeder);

    (life_seeder.worklist, life_seeder.struct_constructors)
}

fn find_live<'tcx>(
    tcx: TyCtxt<'tcx>,
    access_levels: &privacy::AccessLevels,
    krate: &hir::Crate,
) -> FxHashSet<hir::HirId> {
    let (worklist, struct_constructors) = create_and_seed_worklist(tcx, access_levels, krate);
    let mut symbol_visitor = MarkSymbolVisitor {
        worklist,
        tcx,
        tables: &ty::TypeckTables::empty(None),
        live_symbols: Default::default(),
        repr_has_repr_c: false,
        in_pat: false,
        inherited_pub_visibility: false,
        ignore_variant_stack: vec![],
        struct_constructors,
    };
    symbol_visitor.mark_live_symbols();
    symbol_visitor.live_symbols
}

struct DeadVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    live_symbols: FxHashSet<hir::HirId>,
}

impl DeadVisitor<'tcx> {
    fn should_warn_about_item(&mut self, item: &hir::Item) -> bool {
        let should_warn = match item.node {
            hir::ItemKind::Static(..)
            | hir::ItemKind::Const(..)
            | hir::ItemKind::Fn(..)
            | hir::ItemKind::Ty(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..) => true,
            _ => false
        };
        should_warn && !self.symbol_is_live(item.hir_id)
    }

    fn should_warn_about_field(&mut self, field: &hir::StructField) -> bool {
        let field_type = self.tcx.type_of(self.tcx.hir().local_def_id_from_hir_id(field.hir_id));
        !field.is_positional()
            && !self.symbol_is_live(field.hir_id)
            && !field_type.is_phantom_data()
            && !has_allow_dead_code_or_lang_attr(self.tcx, field.hir_id, &field.attrs)
    }

    fn should_warn_about_variant(&mut self, variant: &hir::VariantKind) -> bool {
        !self.symbol_is_live(variant.id)
            && !has_allow_dead_code_or_lang_attr(self.tcx,
                                                 variant.id,
                                                 &variant.attrs)
    }

    fn should_warn_about_foreign_item(&mut self, fi: &hir::ForeignItem) -> bool {
        !self.symbol_is_live(fi.hir_id)
            && !has_allow_dead_code_or_lang_attr(self.tcx, fi.hir_id, &fi.attrs)
    }

    // id := HIR id of an item's definition.
    fn symbol_is_live(
        &mut self,
        id: hir::HirId,
    ) -> bool {
        if self.live_symbols.contains(&id) {
            return true;
        }
        // If it's a type whose items are live, then it's live, too.
        // This is done to handle the case where, for example, the static
        // method of a private type is used, but the type itself is never
        // called directly.
        let def_id = self.tcx.hir().local_def_id_from_hir_id(id);
        let inherent_impls = self.tcx.inherent_impls(def_id);
        for &impl_did in inherent_impls.iter() {
            for &item_did in &self.tcx.associated_item_def_ids(impl_did)[..] {
                if let Some(item_hir_id) = self.tcx.hir().as_local_hir_id(item_did) {
                    if self.live_symbols.contains(&item_hir_id) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn warn_dead_code(&mut self,
                      id: hir::HirId,
                      span: syntax_pos::Span,
                      name: ast::Name,
                      node_type: &str,
                      participle: &str) {
        if !name.as_str().starts_with("_") {
            self.tcx
                .lint_hir(lint::builtin::DEAD_CODE,
                          id,
                          span,
                          &format!("{} is never {}: `{}`",
                                   node_type, participle, name));
        }
    }
}

impl Visitor<'tcx> for DeadVisitor<'tcx> {
    /// Walk nested items in place so that we don't report dead-code
    /// on inner functions when the outer function is already getting
    /// an error. We could do this also by checking the parents, but
    /// this is how the code is setup and it seems harmless enough.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        if self.should_warn_about_item(item) {
            // For items that have a definition with a signature followed by a
            // block, point only at the signature.
            let span = match item.node {
                hir::ItemKind::Fn(..) |
                hir::ItemKind::Mod(..) |
                hir::ItemKind::Enum(..) |
                hir::ItemKind::Struct(..) |
                hir::ItemKind::Union(..) |
                hir::ItemKind::Trait(..) |
                hir::ItemKind::Impl(..) => self.tcx.sess.source_map().def_span(item.span),
                _ => item.span,
            };
            let participle = match item.node {
                hir::ItemKind::Struct(..) => "constructed", // Issue #52325
                _ => "used"
            };
            self.warn_dead_code(
                item.hir_id,
                span,
                item.ident.name,
                item.node.descriptive_variant(),
                participle,
            );
        } else {
            // Only continue if we didn't warn
            intravisit::walk_item(self, item);
        }
    }

    fn visit_variant(&mut self,
                     variant: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     id: hir::HirId) {
        if self.should_warn_about_variant(&variant.node) {
            self.warn_dead_code(variant.node.id, variant.span, variant.node.ident.name,
                                "variant", "constructed");
        } else {
            intravisit::walk_variant(self, variant, g, id);
        }
    }

    fn visit_foreign_item(&mut self, fi: &'tcx hir::ForeignItem) {
        if self.should_warn_about_foreign_item(fi) {
            self.warn_dead_code(fi.hir_id, fi.span, fi.ident.name,
                                fi.node.descriptive_variant(), "used");
        }
        intravisit::walk_foreign_item(self, fi);
    }

    fn visit_struct_field(&mut self, field: &'tcx hir::StructField) {
        if self.should_warn_about_field(&field) {
            self.warn_dead_code(field.hir_id, field.span, field.ident.name, "field", "used");
        }
        intravisit::walk_struct_field(self, field);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        match impl_item.node {
            hir::ImplItemKind::Const(_, body_id) => {
                if !self.symbol_is_live(impl_item.hir_id) {
                    self.warn_dead_code(impl_item.hir_id,
                                        impl_item.span,
                                        impl_item.ident.name,
                                        "associated const",
                                        "used");
                }
                self.visit_nested_body(body_id)
            }
            hir::ImplItemKind::Method(_, body_id) => {
                if !self.symbol_is_live(impl_item.hir_id) {
                    let span = self.tcx.sess.source_map().def_span(impl_item.span);
                    self.warn_dead_code(impl_item.hir_id, span, impl_item.ident.name, "method",
                        "used");
                }
                self.visit_nested_body(body_id)
            }
            hir::ImplItemKind::Existential(..) |
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

pub fn check_crate(tcx: TyCtxt<'_>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);
    let krate = tcx.hir().krate();
    let live_symbols = find_live(tcx, access_levels, krate);
    let mut visitor = DeadVisitor {
        tcx,
        live_symbols,
    };
    intravisit::walk_crate(&mut visitor, krate);
}
