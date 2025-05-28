// The visitors in this module collect sizes and counts of the most important
// pieces of AST and HIR. The resulting numbers are good approximations but not
// completely accurate (some things might be counted twice, others missed).

use rustc_ast::visit::BoundKind;
use rustc_ast::{self as ast, NodeId, visit as ast_visit};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::thousands::format_with_underscores;
use rustc_hir::{self as hir, AmbigArg, HirId, intravisit as hir_visit};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

struct NodeStats {
    count: usize,
    size: usize,
}

impl NodeStats {
    fn new() -> NodeStats {
        NodeStats { count: 0, size: 0 }
    }

    fn accum_size(&self) -> usize {
        self.count * self.size
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
    tcx: Option<TyCtxt<'k>>,
    nodes: FxHashMap<&'static str, Node>,
    seen: FxHashSet<HirId>,
}

pub fn print_hir_stats(tcx: TyCtxt<'_>) {
    let mut collector =
        StatCollector { tcx: Some(tcx), nodes: FxHashMap::default(), seen: FxHashSet::default() };
    tcx.hir_walk_toplevel_module(&mut collector);
    tcx.hir_walk_attributes(&mut collector);
    collector.print("HIR STATS", "hir-stats");
}

pub fn print_ast_stats(krate: &ast::Crate, title: &str, prefix: &str) {
    use rustc_ast::visit::Visitor;

    let mut collector =
        StatCollector { tcx: None, nodes: FxHashMap::default(), seen: FxHashSet::default() };
    collector.visit_crate(krate);
    collector.print(title, prefix);
}

impl<'k> StatCollector<'k> {
    // Record a top-level node.
    fn record<T>(&mut self, label: &'static str, id: Option<HirId>, val: &T) {
        self.record_inner(label, None, id, val);
    }

    // Record a two-level entry, with a top-level enum type and a variant.
    fn record_variant<T>(
        &mut self,
        label1: &'static str,
        label2: &'static str,
        id: Option<HirId>,
        val: &T,
    ) {
        self.record_inner(label1, Some(label2), id, val);
    }

    fn record_inner<T>(
        &mut self,
        label1: &'static str,
        label2: Option<&'static str>,
        id: Option<HirId>,
        val: &T,
    ) {
        if id.is_some_and(|x| !self.seen.insert(x)) {
            return;
        }

        let node = self.nodes.entry(label1).or_insert(Node::new());
        node.stats.count += 1;
        node.stats.size = size_of_val(val);

        if let Some(label2) = label2 {
            let subnode = node.subnodes.entry(label2).or_insert(NodeStats::new());
            subnode.count += 1;
            subnode.size = size_of_val(val);
        }
    }

    fn print(&self, title: &str, prefix: &str) {
        // We will soon sort, so the initial order does not matter.
        #[allow(rustc::potential_query_instability)]
        let mut nodes: Vec<_> = self.nodes.iter().collect();
        nodes.sort_by_cached_key(|(label, node)| (node.stats.accum_size(), label.to_owned()));

        let total_size = nodes.iter().map(|(_, node)| node.stats.accum_size()).sum();
        let total_count = nodes.iter().map(|(_, node)| node.stats.count).sum();

        eprintln!("{prefix} {title}");
        eprintln!(
            "{} {:<18}{:>18}{:>14}{:>14}",
            prefix, "Name", "Accumulated Size", "Count", "Item Size"
        );
        eprintln!("{prefix} ----------------------------------------------------------------");

        let percent = |m, n| (m * 100) as f64 / n as f64;

        for (label, node) in nodes {
            let size = node.stats.accum_size();
            eprintln!(
                "{} {:<18}{:>10} ({:4.1}%){:>14}{:>14}",
                prefix,
                label,
                format_with_underscores(size),
                percent(size, total_size),
                format_with_underscores(node.stats.count),
                format_with_underscores(node.stats.size)
            );
            if !node.subnodes.is_empty() {
                // We will soon sort, so the initial order does not matter.
                #[allow(rustc::potential_query_instability)]
                let mut subnodes: Vec<_> = node.subnodes.iter().collect();
                subnodes.sort_by_cached_key(|(label, subnode)| {
                    (subnode.accum_size(), label.to_owned())
                });

                for (label, subnode) in subnodes {
                    let size = subnode.accum_size();
                    eprintln!(
                        "{} - {:<18}{:>10} ({:4.1}%){:>14}",
                        prefix,
                        label,
                        format_with_underscores(size),
                        percent(size, total_size),
                        format_with_underscores(subnode.count),
                    );
                }
            }
        }
        eprintln!("{prefix} ----------------------------------------------------------------");
        eprintln!(
            "{} {:<18}{:>10}        {:>14}",
            prefix,
            "Total",
            format_with_underscores(total_size),
            format_with_underscores(total_count),
        );
        eprintln!("{prefix}");
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
        self.record("Param", Some(param.hir_id), param);
        hir_visit::walk_param(self, param)
    }

    fn visit_nested_item(&mut self, id: hir::ItemId) {
        let nested_item = self.tcx.unwrap().hir_item(id);
        self.visit_item(nested_item)
    }

    fn visit_nested_trait_item(&mut self, trait_item_id: hir::TraitItemId) {
        let nested_trait_item = self.tcx.unwrap().hir_trait_item(trait_item_id);
        self.visit_trait_item(nested_trait_item)
    }

    fn visit_nested_impl_item(&mut self, impl_item_id: hir::ImplItemId) {
        let nested_impl_item = self.tcx.unwrap().hir_impl_item(impl_item_id);
        self.visit_impl_item(nested_impl_item)
    }

    fn visit_nested_foreign_item(&mut self, id: hir::ForeignItemId) {
        let nested_foreign_item = self.tcx.unwrap().hir_foreign_item(id);
        self.visit_foreign_item(nested_foreign_item);
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let nested_body = self.tcx.unwrap().hir_body(body_id);
        self.visit_body(nested_body)
    }

    fn visit_item(&mut self, i: &'v hir::Item<'v>) {
        record_variants!(
            (self, i, i.kind, Some(i.hir_id()), hir, Item, ItemKind),
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

    fn visit_body(&mut self, b: &hir::Body<'v>) {
        self.record("Body", None, b);
        hir_visit::walk_body(self, b);
    }

    fn visit_mod(&mut self, m: &'v hir::Mod<'v>, _s: Span, _n: HirId) {
        self.record("Mod", None, m);
        hir_visit::walk_mod(self, m)
    }

    fn visit_foreign_item(&mut self, i: &'v hir::ForeignItem<'v>) {
        record_variants!(
            (self, i, i.kind, Some(i.hir_id()), hir, ForeignItem, ForeignItemKind),
            [Fn, Static, Type]
        );
        hir_visit::walk_foreign_item(self, i)
    }

    fn visit_local(&mut self, l: &'v hir::LetStmt<'v>) {
        self.record("Local", Some(l.hir_id), l);
        hir_visit::walk_local(self, l)
    }

    fn visit_block(&mut self, b: &'v hir::Block<'v>) {
        self.record("Block", Some(b.hir_id), b);
        hir_visit::walk_block(self, b)
    }

    fn visit_stmt(&mut self, s: &'v hir::Stmt<'v>) {
        record_variants!(
            (self, s, s.kind, Some(s.hir_id), hir, Stmt, StmtKind),
            [Let, Item, Expr, Semi]
        );
        hir_visit::walk_stmt(self, s)
    }

    fn visit_arm(&mut self, a: &'v hir::Arm<'v>) {
        self.record("Arm", Some(a.hir_id), a);
        hir_visit::walk_arm(self, a)
    }

    fn visit_pat(&mut self, p: &'v hir::Pat<'v>) {
        record_variants!(
            (self, p, p.kind, Some(p.hir_id), hir, Pat, PatKind),
            [
                Missing,
                Wild,
                Binding,
                Struct,
                TupleStruct,
                Or,
                Never,
                Tuple,
                Box,
                Deref,
                Ref,
                Expr,
                Guard,
                Range,
                Slice,
                Err
            ]
        );
        hir_visit::walk_pat(self, p)
    }

    fn visit_pat_field(&mut self, f: &'v hir::PatField<'v>) {
        self.record("PatField", Some(f.hir_id), f);
        hir_visit::walk_pat_field(self, f)
    }

    fn visit_expr(&mut self, e: &'v hir::Expr<'v>) {
        record_variants!(
            (self, e, e.kind, Some(e.hir_id), hir, Expr, ExprKind),
            [
                ConstBlock,
                Array,
                Call,
                MethodCall,
                Use,
                Tup,
                Binary,
                Unary,
                Lit,
                Cast,
                Type,
                DropTemps,
                Let,
                If,
                Loop,
                Match,
                Closure,
                Block,
                Assign,
                AssignOp,
                Field,
                Index,
                Path,
                AddrOf,
                Break,
                Continue,
                Ret,
                Become,
                InlineAsm,
                OffsetOf,
                Struct,
                Repeat,
                Yield,
                UnsafeBinderCast,
                Err
            ]
        );
        hir_visit::walk_expr(self, e)
    }

    fn visit_expr_field(&mut self, f: &'v hir::ExprField<'v>) {
        self.record("ExprField", Some(f.hir_id), f);
        hir_visit::walk_expr_field(self, f)
    }

    fn visit_ty(&mut self, t: &'v hir::Ty<'v, AmbigArg>) {
        record_variants!(
            (self, t, t.kind, Some(t.hir_id), hir, Ty, TyKind),
            [
                InferDelegation,
                Slice,
                Array,
                Ptr,
                Ref,
                BareFn,
                UnsafeBinder,
                Never,
                Tup,
                Path,
                OpaqueDef,
                TraitAscription,
                TraitObject,
                Typeof,
                Infer,
                Pat,
                Err
            ]
        );
        hir_visit::walk_ty(self, t)
    }

    fn visit_generic_param(&mut self, p: &'v hir::GenericParam<'v>) {
        self.record("GenericParam", Some(p.hir_id), p);
        hir_visit::walk_generic_param(self, p)
    }

    fn visit_generics(&mut self, g: &'v hir::Generics<'v>) {
        self.record("Generics", None, g);
        hir_visit::walk_generics(self, g)
    }

    fn visit_where_predicate(&mut self, p: &'v hir::WherePredicate<'v>) {
        record_variants!(
            (self, p, p.kind, Some(p.hir_id), hir, WherePredicate, WherePredicateKind),
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
        self.record("FnDecl", None, fd);
        hir_visit::walk_fn(self, fk, fd, b, id)
    }

    fn visit_use(&mut self, p: &'v hir::UsePath<'v>, _hir_id: HirId) {
        // This is `visit_use`, but the type is `Path` so record it that way.
        self.record("Path", None, p);
        // Don't call `hir_visit::walk_use(self, p, hir_id)`: it calls
        // `visit_path` up to three times, once for each namespace result in
        // `p.res`, by building temporary `Path`s that are not part of the real
        // HIR, which causes `p` to be double- or triple-counted. Instead just
        // walk the path internals (i.e. the segments) directly.
        let hir::Path { span: _, res: _, segments } = *p;
        ast_visit::walk_list!(self, visit_path_segment, segments);
    }

    fn visit_trait_item(&mut self, ti: &'v hir::TraitItem<'v>) {
        record_variants!(
            (self, ti, ti.kind, Some(ti.hir_id()), hir, TraitItem, TraitItemKind),
            [Const, Fn, Type]
        );
        hir_visit::walk_trait_item(self, ti)
    }

    fn visit_trait_item_ref(&mut self, ti: &'v hir::TraitItemRef) {
        self.record("TraitItemRef", Some(ti.id.hir_id()), ti);
        hir_visit::walk_trait_item_ref(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem<'v>) {
        record_variants!(
            (self, ii, ii.kind, Some(ii.hir_id()), hir, ImplItem, ImplItemKind),
            [Const, Fn, Type]
        );
        hir_visit::walk_impl_item(self, ii)
    }

    fn visit_foreign_item_ref(&mut self, fi: &'v hir::ForeignItemRef) {
        self.record("ForeignItemRef", Some(fi.id.hir_id()), fi);
        hir_visit::walk_foreign_item_ref(self, fi)
    }

    fn visit_impl_item_ref(&mut self, ii: &'v hir::ImplItemRef) {
        self.record("ImplItemRef", Some(ii.id.hir_id()), ii);
        hir_visit::walk_impl_item_ref(self, ii)
    }

    fn visit_param_bound(&mut self, b: &'v hir::GenericBound<'v>) {
        record_variants!(
            (self, b, b, None, hir, GenericBound, GenericBound),
            [Trait, Outlives, Use]
        );
        hir_visit::walk_param_bound(self, b)
    }

    fn visit_field_def(&mut self, s: &'v hir::FieldDef<'v>) {
        self.record("FieldDef", Some(s.hir_id), s);
        hir_visit::walk_field_def(self, s)
    }

    fn visit_variant(&mut self, v: &'v hir::Variant<'v>) {
        self.record("Variant", None, v);
        hir_visit::walk_variant(self, v)
    }

    fn visit_generic_arg(&mut self, ga: &'v hir::GenericArg<'v>) {
        record_variants!(
            (self, ga, ga, Some(ga.hir_id()), hir, GenericArg, GenericArg),
            [Lifetime, Type, Const, Infer]
        );
        match ga {
            hir::GenericArg::Lifetime(lt) => self.visit_lifetime(lt),
            hir::GenericArg::Type(ty) => self.visit_ty(ty),
            hir::GenericArg::Const(ct) => self.visit_const_arg(ct),
            hir::GenericArg::Infer(inf) => self.visit_id(inf.hir_id),
        }
    }

    fn visit_lifetime(&mut self, lifetime: &'v hir::Lifetime) {
        self.record("Lifetime", Some(lifetime.hir_id), lifetime);
        hir_visit::walk_lifetime(self, lifetime)
    }

    fn visit_path(&mut self, path: &hir::Path<'v>, _id: HirId) {
        self.record("Path", None, path);
        hir_visit::walk_path(self, path)
    }

    fn visit_path_segment(&mut self, path_segment: &'v hir::PathSegment<'v>) {
        self.record("PathSegment", None, path_segment);
        hir_visit::walk_path_segment(self, path_segment)
    }

    fn visit_generic_args(&mut self, ga: &'v hir::GenericArgs<'v>) {
        self.record("GenericArgs", None, ga);
        hir_visit::walk_generic_args(self, ga)
    }

    fn visit_assoc_item_constraint(&mut self, constraint: &'v hir::AssocItemConstraint<'v>) {
        self.record("AssocItemConstraint", Some(constraint.hir_id), constraint);
        hir_visit::walk_assoc_item_constraint(self, constraint)
    }

    fn visit_attribute(&mut self, attr: &'v hir::Attribute) {
        self.record("Attribute", None, attr);
    }

    fn visit_inline_asm(&mut self, asm: &'v hir::InlineAsm<'v>, id: HirId) {
        self.record("InlineAsm", None, asm);
        hir_visit::walk_inline_asm(self, asm, id);
    }
}

impl<'v> ast_visit::Visitor<'v> for StatCollector<'v> {
    fn visit_foreign_item(&mut self, i: &'v ast::ForeignItem) {
        record_variants!(
            (self, i, i.kind, None, ast, ForeignItem, ForeignItemKind),
            [Static, Fn, TyAlias, MacCall]
        );
        ast_visit::walk_item(self, i)
    }

    fn visit_item(&mut self, i: &'v ast::Item) {
        record_variants!(
            (self, i, i.kind, None, ast, Item, ItemKind),
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
                MacroDef,
                Delegation,
                DelegationMac
            ]
        );
        ast_visit::walk_item(self, i)
    }

    fn visit_local(&mut self, l: &'v ast::Local) {
        self.record("Local", None, l);
        ast_visit::walk_local(self, l)
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        self.record("Block", None, b);
        ast_visit::walk_block(self, b)
    }

    fn visit_stmt(&mut self, s: &'v ast::Stmt) {
        record_variants!(
            (self, s, s.kind, None, ast, Stmt, StmtKind),
            [Let, Item, Expr, Semi, Empty, MacCall]
        );
        ast_visit::walk_stmt(self, s)
    }

    fn visit_param(&mut self, p: &'v ast::Param) {
        self.record("Param", None, p);
        ast_visit::walk_param(self, p)
    }

    fn visit_arm(&mut self, a: &'v ast::Arm) {
        self.record("Arm", None, a);
        ast_visit::walk_arm(self, a)
    }

    fn visit_pat(&mut self, p: &'v ast::Pat) {
        record_variants!(
            (self, p, p.kind, None, ast, Pat, PatKind),
            [
                Missing,
                Wild,
                Ident,
                Struct,
                TupleStruct,
                Or,
                Path,
                Tuple,
                Box,
                Deref,
                Ref,
                Expr,
                Range,
                Slice,
                Rest,
                Never,
                Guard,
                Paren,
                MacCall,
                Err
            ]
        );
        ast_visit::walk_pat(self, p)
    }

    fn visit_expr(&mut self, e: &'v ast::Expr) {
        #[rustfmt::skip]
        record_variants!(
            (self, e, e.kind, None, ast, Expr, ExprKind),
            [
                Array, ConstBlock, Call, MethodCall, Tup, Binary, Unary, Lit, Cast, Type, Let,
                If, While, ForLoop, Loop, Match, Closure, Block, Await, Use, TryBlock, Assign,
                AssignOp, Field, Index, Range, Underscore, Path, AddrOf, Break, Continue, Ret,
                InlineAsm, FormatArgs, OffsetOf, MacCall, Struct, Repeat, Paren, Try, Yield, Yeet,
                Become, IncludedBytes, Gen, UnsafeBinderCast, Err, Dummy
            ]
        );
        ast_visit::walk_expr(self, e)
    }

    fn visit_ty(&mut self, t: &'v ast::Ty) {
        record_variants!(
            (self, t, t.kind, None, ast, Ty, TyKind),
            [
                Slice,
                Array,
                Ptr,
                Ref,
                PinnedRef,
                BareFn,
                UnsafeBinder,
                Never,
                Tup,
                Path,
                Pat,
                TraitObject,
                ImplTrait,
                Paren,
                Typeof,
                Infer,
                ImplicitSelf,
                MacCall,
                CVarArgs,
                Dummy,
                Err
            ]
        );

        ast_visit::walk_ty(self, t)
    }

    fn visit_generic_param(&mut self, g: &'v ast::GenericParam) {
        self.record("GenericParam", None, g);
        ast_visit::walk_generic_param(self, g)
    }

    fn visit_where_predicate(&mut self, p: &'v ast::WherePredicate) {
        record_variants!(
            (self, p, &p.kind, None, ast, WherePredicate, WherePredicateKind),
            [BoundPredicate, RegionPredicate, EqPredicate]
        );
        ast_visit::walk_where_predicate(self, p)
    }

    fn visit_fn(&mut self, fk: ast_visit::FnKind<'v>, _: Span, _: NodeId) {
        self.record("FnDecl", None, fk.decl());
        ast_visit::walk_fn(self, fk)
    }

    fn visit_assoc_item(&mut self, i: &'v ast::AssocItem, ctxt: ast_visit::AssocCtxt) {
        record_variants!(
            (self, i, i.kind, None, ast, AssocItem, AssocItemKind),
            [Const, Fn, Type, MacCall, Delegation, DelegationMac]
        );
        ast_visit::walk_assoc_item(self, i, ctxt);
    }

    fn visit_param_bound(&mut self, b: &'v ast::GenericBound, _ctxt: BoundKind) {
        record_variants!(
            (self, b, b, None, ast, GenericBound, GenericBound),
            [Trait, Outlives, Use]
        );
        ast_visit::walk_param_bound(self, b)
    }

    fn visit_field_def(&mut self, s: &'v ast::FieldDef) {
        self.record("FieldDef", None, s);
        ast_visit::walk_field_def(self, s)
    }

    fn visit_variant(&mut self, v: &'v ast::Variant) {
        self.record("Variant", None, v);
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
        self.record("PathSegment", None, path_segment);
        ast_visit::walk_path_segment(self, path_segment)
    }

    // `GenericArgs` has one inline use (in `ast::AssocItemConstraint::gen_args`) and one
    // non-inline use (in `ast::PathSegment::args`). The latter case is more
    // common, so we implement `visit_generic_args` and tolerate the double
    // counting in the former case.
    fn visit_generic_args(&mut self, g: &'v ast::GenericArgs) {
        record_variants!(
            (self, g, g, None, ast, GenericArgs, GenericArgs),
            [AngleBracketed, Parenthesized, ParenthesizedElided]
        );
        ast_visit::walk_generic_args(self, g)
    }

    fn visit_attribute(&mut self, attr: &'v ast::Attribute) {
        record_variants!(
            (self, attr, attr.kind, None, ast, Attribute, AttrKind),
            [Normal, DocComment]
        );
        ast_visit::walk_attribute(self, attr)
    }

    fn visit_expr_field(&mut self, f: &'v ast::ExprField) {
        self.record("ExprField", None, f);
        ast_visit::walk_expr_field(self, f)
    }

    fn visit_crate(&mut self, krate: &'v ast::Crate) {
        self.record("Crate", None, krate);
        ast_visit::walk_crate(self, krate)
    }

    fn visit_inline_asm(&mut self, asm: &'v ast::InlineAsm) {
        self.record("InlineAsm", None, asm);
        ast_visit::walk_inline_asm(self, asm)
    }
}
