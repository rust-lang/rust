// The visitors in this module collect sizes and counts of the most important
// pieces of AST and HIR. The resulting numbers are good approximations but not
// completely accurate (some things might be counted twice, others missed).

use rustc_ast::visit as ast_visit;
use rustc_ast::{self as ast, AttrId, NodeId};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::intravisit as hir_visit;
use rustc_hir::HirId;
use rustc_middle::hir::map::Map;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::common::to_readable_str;
use rustc_span::Span;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Id {
    Node(HirId),
    Attr(AttrId),
    None,
}

struct NodeData {
    count: usize,
    size: usize,
}

struct StatCollector<'k> {
    krate: Option<Map<'k>>,
    data: FxHashMap<&'static str, NodeData>,
    seen: FxHashSet<Id>,
}

pub fn print_hir_stats(tcx: TyCtxt<'_>) {
    let mut collector = StatCollector {
        krate: Some(tcx.hir()),
        data: FxHashMap::default(),
        seen: FxHashSet::default(),
    };
    tcx.hir().walk_toplevel_module(&mut collector);
    tcx.hir().walk_attributes(&mut collector);
    collector.print("HIR STATS");
}

pub fn print_ast_stats(krate: &ast::Crate, title: &str) {
    let mut collector =
        StatCollector { krate: None, data: FxHashMap::default(), seen: FxHashSet::default() };
    ast_visit::walk_crate(&mut collector, krate);
    collector.print(title);
}

impl<'k> StatCollector<'k> {
    fn record<T>(&mut self, label: &'static str, id: Id, node: &T) {
        if id != Id::None && !self.seen.insert(id) {
            return;
        }

        let entry = self.data.entry(label).or_insert(NodeData { count: 0, size: 0 });

        entry.count += 1;
        entry.size = std::mem::size_of_val(node);
    }

    fn print(&self, title: &str) {
        let mut stats: Vec<_> = self.data.iter().collect();

        stats.sort_by_key(|&(_, ref d)| d.count * d.size);

        let mut total_size = 0;

        eprintln!("\n{}\n", title);

        eprintln!("{:<18}{:>18}{:>14}{:>14}", "Name", "Accumulated Size", "Count", "Item Size");
        eprintln!("----------------------------------------------------------------");

        for (label, data) in stats {
            eprintln!(
                "{:<18}{:>18}{:>14}{:>14}",
                label,
                to_readable_str(data.count * data.size),
                to_readable_str(data.count),
                to_readable_str(data.size)
            );

            total_size += data.count * data.size;
        }
        eprintln!("----------------------------------------------------------------");
        eprintln!("{:<18}{:>18}\n", "Total", to_readable_str(total_size));
    }
}

impl<'v> hir_visit::Visitor<'v> for StatCollector<'v> {
    fn visit_param(&mut self, param: &'v hir::Param<'v>) {
        self.record("Param", Id::Node(param.hir_id), param);
        hir_visit::walk_param(self, param)
    }

    type Map = Map<'v>;

    fn nested_visit_map(&mut self) -> hir_visit::NestedVisitorMap<Self::Map> {
        panic!("visit_nested_xxx must be manually implemented in this visitor")
    }

    fn visit_nested_item(&mut self, id: hir::ItemId) {
        let nested_item = self.krate.unwrap().item(id);
        self.visit_item(nested_item)
    }

    fn visit_nested_trait_item(&mut self, trait_item_id: hir::TraitItemId) {
        let nested_trait_item = self.krate.unwrap().trait_item(trait_item_id);
        self.visit_trait_item(nested_trait_item)
    }

    fn visit_nested_impl_item(&mut self, impl_item_id: hir::ImplItemId) {
        let nested_impl_item = self.krate.unwrap().impl_item(impl_item_id);
        self.visit_impl_item(nested_impl_item)
    }

    fn visit_nested_foreign_item(&mut self, id: hir::ForeignItemId) {
        let nested_foreign_item = self.krate.unwrap().foreign_item(id);
        self.visit_foreign_item(nested_foreign_item);
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let nested_body = self.krate.unwrap().body(body_id);
        self.visit_body(nested_body)
    }

    fn visit_item(&mut self, i: &'v hir::Item<'v>) {
        self.record("Item", Id::Node(i.hir_id()), i);
        hir_visit::walk_item(self, i)
    }

    fn visit_foreign_item(&mut self, i: &'v hir::ForeignItem<'v>) {
        self.record("ForeignItem", Id::Node(i.hir_id()), i);
        hir_visit::walk_foreign_item(self, i)
    }

    fn visit_local(&mut self, l: &'v hir::Local<'v>) {
        self.record("Local", Id::Node(l.hir_id), l);
        hir_visit::walk_local(self, l)
    }

    fn visit_block(&mut self, b: &'v hir::Block<'v>) {
        self.record("Block", Id::Node(b.hir_id), b);
        hir_visit::walk_block(self, b)
    }

    fn visit_stmt(&mut self, s: &'v hir::Stmt<'v>) {
        self.record("Stmt", Id::Node(s.hir_id), s);
        hir_visit::walk_stmt(self, s)
    }

    fn visit_arm(&mut self, a: &'v hir::Arm<'v>) {
        self.record("Arm", Id::Node(a.hir_id), a);
        hir_visit::walk_arm(self, a)
    }

    fn visit_pat(&mut self, p: &'v hir::Pat<'v>) {
        self.record("Pat", Id::Node(p.hir_id), p);
        hir_visit::walk_pat(self, p)
    }

    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        self.record("Expr", Id::Node(ex.hir_id), ex);
        hir_visit::walk_expr(self, ex)
    }

    fn visit_ty(&mut self, t: &'v hir::Ty<'v>) {
        self.record("Ty", Id::Node(t.hir_id), t);
        hir_visit::walk_ty(self, t)
    }

    fn visit_fn(
        &mut self,
        fk: hir_visit::FnKind<'v>,
        fd: &'v hir::FnDecl<'v>,
        b: hir::BodyId,
        s: Span,
        id: hir::HirId,
    ) {
        self.record("FnDecl", Id::None, fd);
        hir_visit::walk_fn(self, fk, fd, b, s, id)
    }

    fn visit_where_predicate(&mut self, predicate: &'v hir::WherePredicate<'v>) {
        self.record("WherePredicate", Id::None, predicate);
        hir_visit::walk_where_predicate(self, predicate)
    }

    fn visit_trait_item(&mut self, ti: &'v hir::TraitItem<'v>) {
        self.record("TraitItem", Id::Node(ti.hir_id()), ti);
        hir_visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem<'v>) {
        self.record("ImplItem", Id::Node(ii.hir_id()), ii);
        hir_visit::walk_impl_item(self, ii)
    }

    fn visit_param_bound(&mut self, bounds: &'v hir::GenericBound<'v>) {
        self.record("GenericBound", Id::None, bounds);
        hir_visit::walk_param_bound(self, bounds)
    }

    fn visit_field_def(&mut self, s: &'v hir::FieldDef<'v>) {
        self.record("FieldDef", Id::Node(s.hir_id), s);
        hir_visit::walk_field_def(self, s)
    }

    fn visit_variant(
        &mut self,
        v: &'v hir::Variant<'v>,
        g: &'v hir::Generics<'v>,
        item_id: hir::HirId,
    ) {
        self.record("Variant", Id::None, v);
        hir_visit::walk_variant(self, v, g, item_id)
    }

    fn visit_lifetime(&mut self, lifetime: &'v hir::Lifetime) {
        self.record("Lifetime", Id::Node(lifetime.hir_id), lifetime);
        hir_visit::walk_lifetime(self, lifetime)
    }

    fn visit_qpath(&mut self, qpath: &'v hir::QPath<'v>, id: hir::HirId, span: Span) {
        self.record("QPath", Id::None, qpath);
        hir_visit::walk_qpath(self, qpath, id, span)
    }

    fn visit_path(&mut self, path: &'v hir::Path<'v>, _id: hir::HirId) {
        self.record("Path", Id::None, path);
        hir_visit::walk_path(self, path)
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'v hir::PathSegment<'v>) {
        self.record("PathSegment", Id::None, path_segment);
        hir_visit::walk_path_segment(self, path_span, path_segment)
    }

    fn visit_assoc_type_binding(&mut self, type_binding: &'v hir::TypeBinding<'v>) {
        self.record("TypeBinding", Id::Node(type_binding.hir_id), type_binding);
        hir_visit::walk_assoc_type_binding(self, type_binding)
    }

    fn visit_attribute(&mut self, _: hir::HirId, attr: &'v ast::Attribute) {
        self.record("Attribute", Id::Attr(attr.id), attr);
    }
}

impl<'v> ast_visit::Visitor<'v> for StatCollector<'v> {
    fn visit_foreign_item(&mut self, i: &'v ast::ForeignItem) {
        self.record("ForeignItem", Id::None, i);
        ast_visit::walk_foreign_item(self, i)
    }

    fn visit_item(&mut self, i: &'v ast::Item) {
        self.record("Item", Id::None, i);
        ast_visit::walk_item(self, i)
    }

    fn visit_local(&mut self, l: &'v ast::Local) {
        self.record("Local", Id::None, l);
        ast_visit::walk_local(self, l)
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        self.record("Block", Id::None, b);
        ast_visit::walk_block(self, b)
    }

    fn visit_stmt(&mut self, s: &'v ast::Stmt) {
        self.record("Stmt", Id::None, s);
        ast_visit::walk_stmt(self, s)
    }

    fn visit_arm(&mut self, a: &'v ast::Arm) {
        self.record("Arm", Id::None, a);
        ast_visit::walk_arm(self, a)
    }

    fn visit_pat(&mut self, p: &'v ast::Pat) {
        self.record("Pat", Id::None, p);
        ast_visit::walk_pat(self, p)
    }

    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        self.record("Expr", Id::None, ex);
        ast_visit::walk_expr(self, ex)
    }

    fn visit_ty(&mut self, t: &'v ast::Ty) {
        self.record("Ty", Id::None, t);
        ast_visit::walk_ty(self, t)
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'v>, s: Span, _: NodeId) {
        self.record("FnDecl", Id::None, fk.decl());
        ast_visit::walk_fn(self, fk, s)
    }

    fn visit_assoc_item(&mut self, item: &'v ast::AssocItem, ctxt: ast_visit::AssocCtxt) {
        let label = match ctxt {
            ast_visit::AssocCtxt::Trait => "TraitItem",
            ast_visit::AssocCtxt::Impl => "ImplItem",
        };
        self.record(label, Id::None, item);
        ast_visit::walk_assoc_item(self, item, ctxt);
    }

    fn visit_param_bound(&mut self, bounds: &'v ast::GenericBound) {
        self.record("GenericBound", Id::None, bounds);
        ast_visit::walk_param_bound(self, bounds)
    }

    fn visit_field_def(&mut self, s: &'v ast::FieldDef) {
        self.record("FieldDef", Id::None, s);
        ast_visit::walk_field_def(self, s)
    }

    fn visit_variant(&mut self, v: &'v ast::Variant) {
        self.record("Variant", Id::None, v);
        ast_visit::walk_variant(self, v)
    }

    fn visit_lifetime(&mut self, lifetime: &'v ast::Lifetime) {
        self.record("Lifetime", Id::None, lifetime);
        ast_visit::walk_lifetime(self, lifetime)
    }

    fn visit_mac_call(&mut self, mac: &'v ast::MacCall) {
        self.record("MacCall", Id::None, mac);
        ast_visit::walk_mac(self, mac)
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'v ast::PathSegment) {
        self.record("PathSegment", Id::None, path_segment);
        ast_visit::walk_path_segment(self, path_span, path_segment)
    }

    fn visit_assoc_ty_constraint(&mut self, constraint: &'v ast::AssocTyConstraint) {
        self.record("AssocTyConstraint", Id::None, constraint);
        ast_visit::walk_assoc_ty_constraint(self, constraint)
    }

    fn visit_attribute(&mut self, attr: &'v ast::Attribute) {
        self.record("Attribute", Id::None, attr);
    }
}
