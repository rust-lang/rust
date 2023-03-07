// The visitors in this module collect sizes and counts of the most important
// pieces of AST and HIR. The resulting numbers are good approximations but not
// completely accurate (some things might be counted twice, others missed).

use rustc_ast::visit as ast_visit;
use rustc_ast::visit::BoundKind;
use rustc_ast::{self as ast, AttrId, NodeId};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::intravisit as hir_visit;
use rustc_hir::HirId;
use rustc_middle::hir::map::Map;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::common::to_readable_str;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Id {
    Node(HirId),
    Attr(AttrId),
    None,
}

struct NodeStats {
    count: usize,
    size: usize,
}

impl NodeStats {
    fn new() -> NodeStats {
        NodeStats { count: 0, size: 0 }
    }
}

struct Node {
    stats: NodeStats,
    subnodes: FxHashMap<&'static str, NodeStats>,
}

impl Node {
    fn new() -> Node {
        Node { stats: NodeStats::new(), subnodes: FxHashMap::default() }
    }
}

/// This type measures the size of AST and HIR nodes, by implementing the AST
/// and HIR `Visitor` traits. But we don't measure every visited type because
/// that could cause double counting.
///
/// For example, `ast::Visitor` has `visit_ident`, but `Ident`s are always
/// stored inline within other AST nodes, so we don't implement `visit_ident`
/// here. In contrast, we do implement `visit_expr` because `ast::Expr` is
/// always stored as `P<ast::Expr>`, and every such expression should be
/// measured separately.
///
/// In general, a `visit_foo` method should be implemented here if the
/// corresponding `Foo` type is always stored on its own, e.g.: `P<Foo>`,
/// `Box<Foo>`, `Vec<Foo>`, `Box<[Foo]>`.
///
/// There are some types in the AST and HIR tree that the visitors do not have
/// a `visit_*` method for, and so we cannot measure these, which is
/// unfortunate.
struct StatCollector<'k> {
    krate: Option<Map<'k>>,
    nodes: FxHashMap<&'static str, Node>,
    seen: FxHashSet<Id>,
}

pub fn print_hir_stats(tcx: TyCtxt<'_>) {
    let mut collector = StatCollector {
        krate: Some(tcx.hir()),
        nodes: FxHashMap::default(),
        seen: FxHashSet::default(),
    };
    tcx.hir().walk_toplevel_module(&mut collector);
    tcx.hir().walk_attributes(&mut collector);
    collector.print("HIR STATS", "hir-stats");
}

pub fn print_ast_stats(krate: &ast::Crate, title: &str, prefix: &str) {
    use rustc_ast::visit::Visitor;

    let mut collector =
        StatCollector { krate: None, nodes: FxHashMap::default(), seen: FxHashSet::default() };
    collector.visit_crate(krate);
    collector.print(title, prefix);
}

impl<'k> StatCollector<'k> {
    // Record a top-level node.
    fn record<T>(&mut self, label: &'static str, id: Id, val: &T) {
        self.record_inner(label, None, id, val);
    }

    // Record a two-level entry, with a top-level enum type and a variant.
    fn record_variant<T>(&mut self, label1: &'static str, label2: &'static str, id: Id, val: &T) {
        self.record_inner(label1, Some(label2), id, val);
    }

    fn record_inner<T>(
        &mut self,
        label1: &'static str,
        label2: Option<&'static str>,
        id: Id,
        val: &T,
    ) {
        if id != Id::None && !self.seen.insert(id) {
            return;
        }

        let node = self.nodes.entry(label1).or_insert(Node::new());
        node.stats.count += 1;
        node.stats.size = std::mem::size_of_val(val);

        if let Some(label2) = label2 {
            let subnode = node.subnodes.entry(label2).or_insert(NodeStats::new());
            subnode.count += 1;
            subnode.size = std::mem::size_of_val(val);
        }
    }

    fn print(&self, title: &str, prefix: &str) {
        let mut nodes: Vec<_> = self.nodes.iter().collect();
        nodes.sort_by_key(|(_, node)| node.stats.count * node.stats.size);

        let total_size = nodes.iter().map(|(_, node)| node.stats.count * node.stats.size).sum();

        eprintln!("{} {}", prefix, title);
        eprintln!(
            "{} {:<18}{:>18}{:>14}{:>14}",
            prefix, "Name", "Accumulated Size", "Count", "Item Size"
        );
        eprintln!("{} ----------------------------------------------------------------", prefix);

        let percent = |m, n| (m * 100) as f64 / n as f64;

        for (label, node) in nodes {
            let size = node.stats.count * node.stats.size;
            eprintln!(
                "{} {:<18}{:>10} ({:4.1}%){:>14}{:>14}",
                prefix,
                label,
                to_readable_str(size),
                percent(size, total_size),
                to_readable_str(node.stats.count),
                to_readable_str(node.stats.size)
            );
            if !node.subnodes.is_empty() {
                let mut subnodes: Vec<_> = node.subnodes.iter().collect();
                subnodes.sort_by_key(|(_, subnode)| subnode.count * subnode.size);

                for (label, subnode) in subnodes {
                    let size = subnode.count * subnode.size;
                    eprintln!(
                        "{} - {:<18}{:>10} ({:4.1}%){:>14}",
                        prefix,
                        label,
                        to_readable_str(size),
                        percent(size, total_size),
                        to_readable_str(subnode.count),
                    );
                }
            }
        }
        eprintln!("{} ----------------------------------------------------------------", prefix);
        eprintln!("{} {:<18}{:>10}", prefix, "Total", to_readable_str(total_size));
        eprintln!("{}", prefix);
    }
}

// Used to avoid boilerplate for types with many variants.
macro_rules! record_variants {
    (
        ($self:ident, $val:expr, $kind:expr, $id:expr, $mod:ident, $ty:ty, $tykind:ident),
        [$($variant:ident),*]
    ) => {
        match $kind {
            $(
                $mod::$tykind::$variant { .. } => {
                    $self.record_variant(stringify!($ty), stringify!($variant), $id, $val)
                }
            )*
        }
    };
}

impl<'v> hir_visit::Visitor<'v> for StatCollector<'v> {
    fn visit_param(&mut self, param: &'v hir::Param<'v>) {
        self.record("Param", Id::Node(param.hir_id), param);
        hir_visit::walk_param(self, param)
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
        record_variants!(
            (self, i, i.kind, Id::Node(i.hir_id()), hir, Item, ItemKind),
            [
                ExternCrate,
                Use,
                Static,
                Const,
                Fn,
                Macro,
                Mod,
                ForeignMod,
                GlobalAsm,
                TyAlias,
                OpaqueTy,
                Enum,
                Struct,
                Union,
                Trait,
                TraitAlias,
                Impl
            ]
        );
        hir_visit::walk_item(self, i)
    }

    fn visit_body(&mut self, b: &'v hir::Body<'v>) {
        self.record("Body", Id::None, b);
        hir_visit::walk_body(self, b);
    }

    fn visit_mod(&mut self, m: &'v hir::Mod<'v>, _s: Span, n: HirId) {
        self.record("Mod", Id::None, m);
        hir_visit::walk_mod(self, m, n)
    }

    fn visit_foreign_item(&mut self, i: &'v hir::ForeignItem<'v>) {
        record_variants!(
            (self, i, i.kind, Id::Node(i.hir_id()), hir, ForeignItem, ForeignItemKind),
            [Fn, Static, Type]
        );
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
        record_variants!(
            (self, s, s.kind, Id::Node(s.hir_id), hir, Stmt, StmtKind),
            [Local, Item, Expr, Semi]
        );
        hir_visit::walk_stmt(self, s)
    }

    fn visit_arm(&mut self, a: &'v hir::Arm<'v>) {
        self.record("Arm", Id::Node(a.hir_id), a);
        hir_visit::walk_arm(self, a)
    }

    fn visit_pat(&mut self, p: &'v hir::Pat<'v>) {
        record_variants!(
            (self, p, p.kind, Id::Node(p.hir_id), hir, Pat, PatKind),
            [Wild, Binding, Struct, TupleStruct, Or, Path, Tuple, Box, Ref, Lit, Range, Slice]
        );
        hir_visit::walk_pat(self, p)
    }

    fn visit_pat_field(&mut self, f: &'v hir::PatField<'v>) {
        self.record("PatField", Id::Node(f.hir_id), f);
        hir_visit::walk_pat_field(self, f)
    }

    fn visit_expr(&mut self, e: &'v hir::Expr<'v>) {
        record_variants!(
            (self, e, e.kind, Id::Node(e.hir_id), hir, Expr, ExprKind),
            [
                Box, ConstBlock, Array, Call, MethodCall, Tup, Binary, Unary, Lit, Cast, Type,
                DropTemps, Let, If, Loop, Match, Closure, Block, Assign, AssignOp, Field, Index,
                Path, AddrOf, Break, Continue, Ret, InlineAsm, Struct, Repeat, Yield, Err
            ]
        );
        hir_visit::walk_expr(self, e)
    }

    fn visit_let_expr(&mut self, lex: &'v hir::Let<'v>) {
        self.record("Let", Id::Node(lex.hir_id), lex);
        hir_visit::walk_let_expr(self, lex)
    }

    fn visit_expr_field(&mut self, f: &'v hir::ExprField<'v>) {
        self.record("ExprField", Id::Node(f.hir_id), f);
        hir_visit::walk_expr_field(self, f)
    }

    fn visit_ty(&mut self, t: &'v hir::Ty<'v>) {
        record_variants!(
            (self, t, t.kind, Id::Node(t.hir_id), hir, Ty, TyKind),
            [
                Slice,
                Array,
                Ptr,
                Ref,
                BareFn,
                Never,
                Tup,
                Path,
                OpaqueDef,
                TraitObject,
                Typeof,
                Infer,
                Err
            ]
        );
        hir_visit::walk_ty(self, t)
    }

    fn visit_generic_param(&mut self, p: &'v hir::GenericParam<'v>) {
        self.record("GenericParam", Id::Node(p.hir_id), p);
        hir_visit::walk_generic_param(self, p)
    }

    fn visit_generics(&mut self, g: &'v hir::Generics<'v>) {
        self.record("Generics", Id::None, g);
        hir_visit::walk_generics(self, g)
    }

    fn visit_where_predicate(&mut self, p: &'v hir::WherePredicate<'v>) {
        record_variants!(
            (self, p, p, Id::None, hir, WherePredicate, WherePredicate),
            [BoundPredicate, RegionPredicate, EqPredicate]
        );
        hir_visit::walk_where_predicate(self, p)
    }

    fn visit_fn(
        &mut self,
        fk: hir_visit::FnKind<'v>,
        fd: &'v hir::FnDecl<'v>,
        b: hir::BodyId,
        _: Span,
        id: LocalDefId,
    ) {
        self.record("FnDecl", Id::None, fd);
        hir_visit::walk_fn(self, fk, fd, b, id)
    }

    fn visit_use(&mut self, p: &'v hir::UsePath<'v>, hir_id: hir::HirId) {
        // This is `visit_use`, but the type is `Path` so record it that way.
        self.record("Path", Id::None, p);
        hir_visit::walk_use(self, p, hir_id)
    }

    fn visit_trait_item(&mut self, ti: &'v hir::TraitItem<'v>) {
        record_variants!(
            (self, ti, ti.kind, Id::Node(ti.hir_id()), hir, TraitItem, TraitItemKind),
            [Const, Fn, Type]
        );
        hir_visit::walk_trait_item(self, ti)
    }

    fn visit_trait_item_ref(&mut self, ti: &'v hir::TraitItemRef) {
        self.record("TraitItemRef", Id::Node(ti.id.hir_id()), ti);
        hir_visit::walk_trait_item_ref(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem<'v>) {
        record_variants!(
            (self, ii, ii.kind, Id::Node(ii.hir_id()), hir, ImplItem, ImplItemKind),
            [Const, Fn, Type]
        );
        hir_visit::walk_impl_item(self, ii)
    }

    fn visit_foreign_item_ref(&mut self, fi: &'v hir::ForeignItemRef) {
        self.record("ForeignItemRef", Id::Node(fi.id.hir_id()), fi);
        hir_visit::walk_foreign_item_ref(self, fi)
    }

    fn visit_impl_item_ref(&mut self, ii: &'v hir::ImplItemRef) {
        self.record("ImplItemRef", Id::Node(ii.id.hir_id()), ii);
        hir_visit::walk_impl_item_ref(self, ii)
    }

    fn visit_param_bound(&mut self, b: &'v hir::GenericBound<'v>) {
        record_variants!(
            (self, b, b, Id::None, hir, GenericBound, GenericBound),
            [Trait, LangItemTrait, Outlives]
        );
        hir_visit::walk_param_bound(self, b)
    }

    fn visit_field_def(&mut self, s: &'v hir::FieldDef<'v>) {
        self.record("FieldDef", Id::Node(s.hir_id), s);
        hir_visit::walk_field_def(self, s)
    }

    fn visit_variant(&mut self, v: &'v hir::Variant<'v>) {
        self.record("Variant", Id::None, v);
        hir_visit::walk_variant(self, v)
    }

    fn visit_generic_arg(&mut self, ga: &'v hir::GenericArg<'v>) {
        record_variants!(
            (self, ga, ga, Id::Node(ga.hir_id()), hir, GenericArg, GenericArg),
            [Lifetime, Type, Const, Infer]
        );
        match ga {
            hir::GenericArg::Lifetime(lt) => self.visit_lifetime(lt),
            hir::GenericArg::Type(ty) => self.visit_ty(ty),
            hir::GenericArg::Const(ct) => self.visit_anon_const(&ct.value),
            hir::GenericArg::Infer(inf) => self.visit_infer(inf),
        }
    }

    fn visit_lifetime(&mut self, lifetime: &'v hir::Lifetime) {
        self.record("Lifetime", Id::Node(lifetime.hir_id), lifetime);
        hir_visit::walk_lifetime(self, lifetime)
    }

    fn visit_path(&mut self, path: &hir::Path<'v>, _id: hir::HirId) {
        self.record("Path", Id::None, path);
        hir_visit::walk_path(self, path)
    }

    fn visit_path_segment(&mut self, path_segment: &'v hir::PathSegment<'v>) {
        self.record("PathSegment", Id::None, path_segment);
        hir_visit::walk_path_segment(self, path_segment)
    }

    fn visit_generic_args(&mut self, ga: &'v hir::GenericArgs<'v>) {
        self.record("GenericArgs", Id::None, ga);
        hir_visit::walk_generic_args(self, ga)
    }

    fn visit_assoc_type_binding(&mut self, type_binding: &'v hir::TypeBinding<'v>) {
        self.record("TypeBinding", Id::Node(type_binding.hir_id), type_binding);
        hir_visit::walk_assoc_type_binding(self, type_binding)
    }

    fn visit_attribute(&mut self, attr: &'v ast::Attribute) {
        self.record("Attribute", Id::Attr(attr.id), attr);
    }

    fn visit_inline_asm(&mut self, asm: &'v hir::InlineAsm<'v>, id: HirId) {
        self.record("InlineAsm", Id::None, asm);
        hir_visit::walk_inline_asm(self, asm, id);
    }
}

impl<'v> ast_visit::Visitor<'v> for StatCollector<'v> {
    fn visit_foreign_item(&mut self, i: &'v ast::ForeignItem) {
        record_variants!(
            (self, i, i.kind, Id::None, ast, ForeignItem, ForeignItemKind),
            [Static, Fn, TyAlias, MacCall]
        );
        ast_visit::walk_foreign_item(self, i)
    }

    fn visit_item(&mut self, i: &'v ast::Item) {
        record_variants!(
            (self, i, i.kind, Id::None, ast, Item, ItemKind),
            [
                ExternCrate,
                Use,
                Static,
                Const,
                Fn,
                Mod,
                ForeignMod,
                GlobalAsm,
                TyAlias,
                Enum,
                Struct,
                Union,
                Trait,
                TraitAlias,
                Impl,
                MacCall,
                MacroDef
            ]
        );
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
        record_variants!(
            (self, s, s.kind, Id::None, ast, Stmt, StmtKind),
            [Local, Item, Expr, Semi, Empty, MacCall]
        );
        ast_visit::walk_stmt(self, s)
    }

    fn visit_param(&mut self, p: &'v ast::Param) {
        self.record("Param", Id::None, p);
        ast_visit::walk_param(self, p)
    }

    fn visit_arm(&mut self, a: &'v ast::Arm) {
        self.record("Arm", Id::None, a);
        ast_visit::walk_arm(self, a)
    }

    fn visit_pat(&mut self, p: &'v ast::Pat) {
        record_variants!(
            (self, p, p.kind, Id::None, ast, Pat, PatKind),
            [
                Wild,
                Ident,
                Struct,
                TupleStruct,
                Or,
                Path,
                Tuple,
                Box,
                Ref,
                Lit,
                Range,
                Slice,
                Rest,
                Paren,
                MacCall
            ]
        );
        ast_visit::walk_pat(self, p)
    }

    fn visit_expr(&mut self, e: &'v ast::Expr) {
        #[rustfmt::skip]
        record_variants!(
            (self, e, e.kind, Id::None, ast, Expr, ExprKind),
            [
                Box, Array, ConstBlock, Call, MethodCall, Tup, Binary, Unary, Lit, Cast, Type, Let,
                If, While, ForLoop, Loop, Match, Closure, Block, Async, Await, TryBlock, Assign,
                AssignOp, Field, Index, Range, Underscore, Path, AddrOf, Break, Continue, Ret,
                InlineAsm, FormatArgs, MacCall, Struct, Repeat, Paren, Try, Yield, Yeet, IncludedBytes, Err
            ]
        );
        ast_visit::walk_expr(self, e)
    }

    fn visit_ty(&mut self, t: &'v ast::Ty) {
        record_variants!(
            (self, t, t.kind, Id::None, ast, Ty, TyKind),
            [
                Slice,
                Array,
                Ptr,
                Ref,
                BareFn,
                Never,
                Tup,
                Path,
                TraitObject,
                ImplTrait,
                Paren,
                Typeof,
                Infer,
                ImplicitSelf,
                MacCall,
                Err,
                CVarArgs
            ]
        );

        ast_visit::walk_ty(self, t)
    }

    fn visit_generic_param(&mut self, g: &'v ast::GenericParam) {
        self.record("GenericParam", Id::None, g);
        ast_visit::walk_generic_param(self, g)
    }

    fn visit_where_predicate(&mut self, p: &'v ast::WherePredicate) {
        record_variants!(
            (self, p, p, Id::None, ast, WherePredicate, WherePredicate),
            [BoundPredicate, RegionPredicate, EqPredicate]
        );
        ast_visit::walk_where_predicate(self, p)
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'v>, _: Span, _: NodeId) {
        self.record("FnDecl", Id::None, fk.decl());
        ast_visit::walk_fn(self, fk)
    }

    fn visit_assoc_item(&mut self, i: &'v ast::AssocItem, ctxt: ast_visit::AssocCtxt) {
        record_variants!(
            (self, i, i.kind, Id::None, ast, AssocItem, AssocItemKind),
            [Const, Fn, Type, MacCall]
        );
        ast_visit::walk_assoc_item(self, i, ctxt);
    }

    fn visit_param_bound(&mut self, b: &'v ast::GenericBound, _ctxt: BoundKind) {
        record_variants!(
            (self, b, b, Id::None, ast, GenericBound, GenericBound),
            [Trait, Outlives]
        );
        ast_visit::walk_param_bound(self, b)
    }

    fn visit_field_def(&mut self, s: &'v ast::FieldDef) {
        self.record("FieldDef", Id::None, s);
        ast_visit::walk_field_def(self, s)
    }

    fn visit_variant(&mut self, v: &'v ast::Variant) {
        self.record("Variant", Id::None, v);
        ast_visit::walk_variant(self, v)
    }

    // `UseTree` has one inline use (in `ast::ItemKind::Use`) and one
    // non-inline use (in `ast::UseTreeKind::Nested`). The former case is more
    // common, so we don't implement `visit_use_tree` and tolerate the missed
    // coverage in the latter case.

    // `PathSegment` has one inline use (in `ast::ExprKind::MethodCall`) and
    // one non-inline use (in `ast::Path::segments`). The latter case is more
    // common than the former case, so we implement this visitor and tolerate
    // the double counting in the former case.
    fn visit_path_segment(&mut self, path_segment: &'v ast::PathSegment) {
        self.record("PathSegment", Id::None, path_segment);
        ast_visit::walk_path_segment(self, path_segment)
    }

    // `GenericArgs` has one inline use (in `ast::AssocConstraint::gen_args`) and one
    // non-inline use (in `ast::PathSegment::args`). The latter case is more
    // common, so we implement `visit_generic_args` and tolerate the double
    // counting in the former case.
    fn visit_generic_args(&mut self, g: &'v ast::GenericArgs) {
        record_variants!(
            (self, g, g, Id::None, ast, GenericArgs, GenericArgs),
            [AngleBracketed, Parenthesized]
        );
        ast_visit::walk_generic_args(self, g)
    }

    fn visit_attribute(&mut self, attr: &'v ast::Attribute) {
        record_variants!(
            (self, attr, attr.kind, Id::None, ast, Attribute, AttrKind),
            [Normal, DocComment]
        );
        ast_visit::walk_attribute(self, attr)
    }

    fn visit_expr_field(&mut self, f: &'v ast::ExprField) {
        self.record("ExprField", Id::None, f);
        ast_visit::walk_expr_field(self, f)
    }

    fn visit_crate(&mut self, krate: &'v ast::Crate) {
        self.record("Crate", Id::None, krate);
        ast_visit::walk_crate(self, krate)
    }

    fn visit_inline_asm(&mut self, asm: &'v ast::InlineAsm) {
        self.record("InlineAsm", Id::None, asm);
        ast_visit::walk_inline_asm(self, asm)
    }
}
