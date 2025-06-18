//! AST walker. Each overridden visit method has full control over what
//! happens with its node, it can do its own traversal of the node's children,
//! call `visit::walk_*` to apply the default traversal algorithm, or prevent
//! deeper traversal by doing nothing.
//!
//! Note: it is an important invariant that the default visitor walks the body
//! of a function in "execution order" (more concretely, reverse post-order
//! with respect to the CFG implied by the AST), meaning that if AST node A may
//! execute before AST node B, then A is visited first. The borrow checker in
//! particular relies on this property.
//!
//! Note: walking an AST before macro expansion is probably a bad idea. For
//! instance, a walker looking for item names in a module will miss all of
//! those that are created by the expansion of a macro.

pub use rustc_ast_ir::visit::VisitorResult;
pub use rustc_ast_ir::{try_visit, visit_opt, walk_list, walk_visitable_list};
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::ptr::P;
use crate::tokenstream::DelimSpan;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssocCtxt {
    Trait,
    Impl { of_trait: bool },
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FnCtxt {
    Free,
    Foreign,
    Assoc(AssocCtxt),
}

#[derive(Copy, Clone, Debug)]
pub enum BoundKind {
    /// Trait bounds in generics bounds and type/trait alias.
    /// E.g., `<T: Bound>`, `type A: Bound`, or `where T: Bound`.
    Bound,

    /// Trait bounds in `impl` type.
    /// E.g., `type Foo = impl Bound1 + Bound2 + Bound3`.
    Impl,

    /// Trait bounds in trait object type.
    /// E.g., `dyn Bound1 + Bound2 + Bound3`.
    TraitObject,

    /// Super traits of a trait.
    /// E.g., `trait A: B`
    SuperTraits,
}
impl BoundKind {
    pub fn descr(self) -> &'static str {
        match self {
            BoundKind::Bound => "bounds",
            BoundKind::Impl => "`impl Trait`",
            BoundKind::TraitObject => "`dyn` trait object bounds",
            BoundKind::SuperTraits => "supertrait bounds",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FnKind<'a> {
    /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
    Fn(FnCtxt, &'a Visibility, &'a Fn),

    /// E.g., `|x, y| body`.
    Closure(&'a ClosureBinder, &'a Option<CoroutineKind>, &'a FnDecl, &'a Expr),
}

impl<'a> FnKind<'a> {
    pub fn header(&self) -> Option<&'a FnHeader> {
        match *self {
            FnKind::Fn(_, _, Fn { sig, .. }) => Some(&sig.header),
            FnKind::Closure(..) => None,
        }
    }

    pub fn ident(&self) -> Option<&Ident> {
        match self {
            FnKind::Fn(_, _, Fn { ident, .. }) => Some(ident),
            _ => None,
        }
    }

    pub fn decl(&self) -> &'a FnDecl {
        match self {
            FnKind::Fn(_, _, Fn { sig, .. }) => &sig.decl,
            FnKind::Closure(_, _, decl, _) => decl,
        }
    }

    pub fn ctxt(&self) -> Option<FnCtxt> {
        match self {
            FnKind::Fn(ctxt, ..) => Some(*ctxt),
            FnKind::Closure(..) => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LifetimeCtxt {
    /// Appears in a reference type.
    Ref,
    /// Appears as a bound on a type or another lifetime.
    Bound,
    /// Appears as a generic argument.
    GenericArg,
}

/// Each method of the `Visitor` trait is a hook to be potentially
/// overridden. Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_item` method by default calls `visit::walk_item`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method. (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
///
/// Every `walk_*` method uses deconstruction to access fields of structs and
/// enums. This will result in a compile error if a field is added, which makes
/// it more likely the appropriate visit call will be added for it.
pub trait Visitor<'ast>: Sized {
    /// The result type of the `visit_*` methods. Can be either `()`,
    /// or `ControlFlow<T>`.
    type Result: VisitorResult = ();

    fn visit_ident(&mut self, _ident: &'ast Ident) -> Self::Result {
        Self::Result::output()
    }
    fn visit_foreign_mod(&mut self, nm: &'ast ForeignMod) -> Self::Result {
        walk_foreign_mod(self, nm)
    }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) -> Self::Result {
        walk_item(self, i)
    }
    fn visit_item(&mut self, i: &'ast Item) -> Self::Result {
        walk_item(self, i)
    }
    fn visit_local(&mut self, l: &'ast Local) -> Self::Result {
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'ast Block) -> Self::Result {
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'ast Stmt) -> Self::Result {
        walk_stmt(self, s)
    }
    fn visit_param(&mut self, param: &'ast Param) -> Self::Result {
        walk_param(self, param)
    }
    fn visit_arm(&mut self, a: &'ast Arm) -> Self::Result {
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'ast Pat) -> Self::Result {
        walk_pat(self, p)
    }
    fn visit_anon_const(&mut self, c: &'ast AnonConst) -> Self::Result {
        walk_anon_const(self, c)
    }
    fn visit_expr(&mut self, ex: &'ast Expr) -> Self::Result {
        walk_expr(self, ex)
    }
    /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
    /// It can be removed once that feature is stabilized.
    fn visit_method_receiver_expr(&mut self, ex: &'ast Expr) -> Self::Result {
        self.visit_expr(ex)
    }
    fn visit_ty(&mut self, t: &'ast Ty) -> Self::Result {
        walk_ty(self, t)
    }
    fn visit_ty_pat(&mut self, t: &'ast TyPat) -> Self::Result {
        walk_ty_pat(self, t)
    }
    fn visit_generic_param(&mut self, param: &'ast GenericParam) -> Self::Result {
        walk_generic_param(self, param)
    }
    fn visit_generics(&mut self, g: &'ast Generics) -> Self::Result {
        walk_generics(self, g)
    }
    fn visit_closure_binder(&mut self, b: &'ast ClosureBinder) -> Self::Result {
        walk_closure_binder(self, b)
    }
    fn visit_contract(&mut self, c: &'ast FnContract) -> Self::Result {
        walk_contract(self, c)
    }
    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) -> Self::Result {
        walk_where_predicate(self, p)
    }
    fn visit_where_predicate_kind(&mut self, k: &'ast WherePredicateKind) -> Self::Result {
        walk_where_predicate_kind(self, k)
    }
    fn visit_fn(&mut self, fk: FnKind<'ast>, _: Span, _: NodeId) -> Self::Result {
        walk_fn(self, fk)
    }
    fn visit_assoc_item(&mut self, i: &'ast AssocItem, ctxt: AssocCtxt) -> Self::Result {
        walk_assoc_item(self, i, ctxt)
    }
    fn visit_trait_ref(&mut self, t: &'ast TraitRef) -> Self::Result {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'ast GenericBound, _ctxt: BoundKind) -> Self::Result {
        walk_param_bound(self, bounds)
    }
    fn visit_precise_capturing_arg(&mut self, arg: &'ast PreciseCapturingArg) -> Self::Result {
        walk_precise_capturing_arg(self, arg)
    }
    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef) -> Self::Result {
        walk_poly_trait_ref(self, t)
    }
    fn visit_variant_data(&mut self, s: &'ast VariantData) -> Self::Result {
        walk_variant_data(self, s)
    }
    fn visit_field_def(&mut self, s: &'ast FieldDef) -> Self::Result {
        walk_field_def(self, s)
    }
    fn visit_variant(&mut self, v: &'ast Variant) -> Self::Result {
        walk_variant(self, v)
    }
    fn visit_variant_discr(&mut self, discr: &'ast AnonConst) -> Self::Result {
        self.visit_anon_const(discr)
    }
    fn visit_label(&mut self, label: &'ast Label) -> Self::Result {
        walk_label(self, label)
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) -> Self::Result {
        walk_lifetime(self, lifetime)
    }
    fn visit_mac_call(&mut self, mac: &'ast MacCall) -> Self::Result {
        walk_mac(self, mac)
    }
    fn visit_id(&mut self, _id: NodeId) -> Self::Result {
        Self::Result::output()
    }
    fn visit_macro_def(&mut self, macro_def: &'ast MacroDef) -> Self::Result {
        walk_macro_def(self, macro_def)
    }
    fn visit_path(&mut self, path: &'ast Path) -> Self::Result {
        walk_path(self, path)
    }
    fn visit_use_tree(&mut self, use_tree: &'ast UseTree) -> Self::Result {
        walk_use_tree(self, use_tree)
    }
    fn visit_nested_use_tree(&mut self, use_tree: &'ast UseTree, id: NodeId) -> Self::Result {
        try_visit!(self.visit_id(id));
        self.visit_use_tree(use_tree)
    }
    fn visit_path_segment(&mut self, path_segment: &'ast PathSegment) -> Self::Result {
        walk_path_segment(self, path_segment)
    }
    fn visit_generic_args(&mut self, generic_args: &'ast GenericArgs) -> Self::Result {
        walk_generic_args(self, generic_args)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'ast GenericArg) -> Self::Result {
        walk_generic_arg(self, generic_arg)
    }
    fn visit_assoc_item_constraint(
        &mut self,
        constraint: &'ast AssocItemConstraint,
    ) -> Self::Result {
        walk_assoc_item_constraint(self, constraint)
    }
    fn visit_attribute(&mut self, attr: &'ast Attribute) -> Self::Result {
        walk_attribute(self, attr)
    }
    fn visit_vis(&mut self, vis: &'ast Visibility) -> Self::Result {
        walk_vis(self, vis)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'ast FnRetTy) -> Self::Result {
        walk_fn_ret_ty(self, ret_ty)
    }
    fn visit_fn_header(&mut self, header: &'ast FnHeader) -> Self::Result {
        walk_fn_header(self, header)
    }
    fn visit_expr_field(&mut self, f: &'ast ExprField) -> Self::Result {
        walk_expr_field(self, f)
    }
    fn visit_pat_field(&mut self, fp: &'ast PatField) -> Self::Result {
        walk_pat_field(self, fp)
    }
    fn visit_crate(&mut self, krate: &'ast Crate) -> Self::Result {
        walk_crate(self, krate)
    }
    fn visit_inline_asm(&mut self, asm: &'ast InlineAsm) -> Self::Result {
        walk_inline_asm(self, asm)
    }
    fn visit_format_args(&mut self, fmt: &'ast FormatArgs) -> Self::Result {
        walk_format_args(self, fmt)
    }
    fn visit_inline_asm_sym(&mut self, sym: &'ast InlineAsmSym) -> Self::Result {
        walk_inline_asm_sym(self, sym)
    }
    fn visit_capture_by(&mut self, _capture_by: &'ast CaptureBy) -> Self::Result {
        Self::Result::output()
    }
    fn visit_coroutine_kind(&mut self, coroutine_kind: &'ast CoroutineKind) -> Self::Result {
        walk_coroutine_kind(self, coroutine_kind)
    }
    fn visit_fn_decl(&mut self, fn_decl: &'ast FnDecl) -> Self::Result {
        walk_fn_decl(self, fn_decl)
    }
    fn visit_qself(&mut self, qs: &'ast Option<P<QSelf>>) -> Self::Result {
        walk_qself(self, qs)
    }
}

#[macro_export]
macro_rules! common_visitor_and_walkers {
    ($(($mut: ident))? $Visitor:ident$(<$lt:lifetime>)?) => {
        pub trait WalkItemKind {
            type Ctxt;
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result;
        }

        // this is only used by the MutVisitor. We include this symmetry here to make writing other functions easier
        $(${ignore($lt)}
            #[expect(unused, rustc::pass_by_value)]
            #[inline]
        )?
        fn visit_span<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, span: &$($lt)? $($mut)? Span) -> V::Result {
            $(
                ${ignore($mut)}
                vis.visit_span(span);
            )?
            V::Result::output()
        }

        /// helper since `Visitor` wants `NodeId` but `MutVisitor` wants `&mut NodeId`
        $(${ignore($lt)}
            #[expect(rustc::pass_by_value)]
        )?
        #[inline]
        fn visit_id<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, id: &$($lt)? $($mut)? NodeId) -> V::Result {
            // deref `&NodeId` into `NodeId` only for `Visitor`
            vis.visit_id( $(${ignore($lt)} * )? id)
        }

        // this is only used by the MutVisitor. We include this symmetry here to make writing other functions easier
        fn visit_safety<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, safety: &$($lt)? $($mut)? Safety) -> V::Result {
            match safety {
                Safety::Unsafe(span) => visit_span(vis, span),
                Safety::Safe(span) => visit_span(vis, span),
                Safety::Default => { V::Result::output() }
            }
        }

        fn visit_constness<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, constness: &$($lt)? $($mut)? Const) -> V::Result {
            match constness {
                Const::Yes(span) => visit_span(vis, span),
                Const::No => {
                    V::Result::output()
                }
            }
        }

        fn visit_defaultness<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, defaultness: &$($lt)? $($mut)? Defaultness) -> V::Result {
            match defaultness {
                Defaultness::Default(span) => visit_span(vis, span),
                Defaultness::Final => {
                    V::Result::output()
                }
            }
        }

        fn visit_polarity<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            polarity: &$($lt)? $($mut)? ImplPolarity,
        ) -> V::Result {
            match polarity {
                ImplPolarity::Positive => { V::Result::output() }
                ImplPolarity::Negative(span) => visit_span(vis, span),
            }
        }

        $(${ignore($lt)}
            #[inline]
        )?
        fn visit_modifiers<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            m: &$($lt)? $($mut)? TraitBoundModifiers
        ) -> V::Result {
            let TraitBoundModifiers { constness, asyncness, polarity } = m;
            match constness {
                BoundConstness::Never => {}
                BoundConstness::Always(span) | BoundConstness::Maybe(span) => try_visit!(visit_span(vis, span)),
            }
            match asyncness {
                BoundAsyncness::Normal => {}
                BoundAsyncness::Async(span) => try_visit!(visit_span(vis, span)),
            }
            match polarity {
                BoundPolarity::Positive => {}
                BoundPolarity::Negative(span) | BoundPolarity::Maybe(span) => try_visit!(visit_span(vis, span)),
            }
            V::Result::output()
        }

        fn visit_bounds<$($lt,)? V: $Visitor$(<$lt>)?>(visitor: &mut V, bounds: &$($lt)? $($mut)? GenericBounds, ctxt: BoundKind) -> V::Result {
            walk_list!(visitor, visit_param_bound, bounds, ctxt);
            V::Result::output()
        }

        pub fn walk_label<$($lt,)? V: $Visitor$(<$lt>)?>(visitor: &mut V, Label { ident }: &$($lt)? $($mut)? Label) -> V::Result {
            visitor.visit_ident(ident)
        }

        pub fn walk_fn_header<$($lt,)? V: $Visitor$(<$lt>)?>(visitor: &mut V, header: &$($lt)? $($mut)? FnHeader) -> V::Result {
            let FnHeader { safety, coroutine_kind, constness, ext: _ } = header;
            try_visit!(visit_constness(visitor, constness));
            visit_opt!(visitor, visit_coroutine_kind, coroutine_kind);
            visit_safety(visitor, safety)
        }

        pub fn walk_lifetime<$($lt,)? V: $Visitor$(<$lt>)?>(visitor: &mut V, Lifetime { id, ident }: &$($lt)? $($mut)? Lifetime) -> V::Result {
            try_visit!(visit_id(visitor, id));
            visitor.visit_ident(ident)
        }

        fn walk_item_ctxt<$($lt,)? V: $Visitor$(<$lt>)?, K: WalkItemKind>(
            visitor: &mut V,
            item: &$($mut)? $($lt)? Item<K>,
            ctxt: K::Ctxt,
        ) -> V::Result {
            let Item { attrs, id, kind, vis, span, tokens: _ } = item;
            try_visit!(visit_id(visitor, id));
            walk_list!(visitor, visit_attribute, attrs);
            try_visit!(visitor.visit_vis(vis));
            try_visit!(kind.walk(*span, *id, vis, ctxt, visitor));
            visit_span(visitor, span)
        }

        pub fn walk_item<$($lt,)? V: $Visitor$(<$lt>)?, K: WalkItemKind<Ctxt = ()>>(
            visitor: &mut V,
            item: &$($mut)? $($lt)? Item<K>,
        ) -> V::Result {
            walk_item_ctxt(visitor, item, ())
        }

        pub fn walk_assoc_item<$($lt,)? V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            item: &$($mut)? $($lt)? AssocItem,
            ctxt: AssocCtxt,
        ) -> V::Result {
            walk_item_ctxt(visitor, item, ctxt)
        }

        impl WalkItemKind for ItemKind {
            type Ctxt = ();
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                _ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    ItemKind::ExternCrate(_orig_name, ident) => vis.visit_ident(ident),
                    ItemKind::Use(use_tree) => vis.visit_use_tree(use_tree),
                    ItemKind::Static(box StaticItem {
                        ident,
                        ty,
                        safety: _,
                        mutability: _,
                        expr,
                        define_opaque,
                    }) => {
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_ty(ty));
                        visit_opt!(vis, visit_expr, expr);
                        walk_define_opaques(vis, define_opaque)
                    }
                    ItemKind::Const(item) => {
                        walk_const_item(vis, item)
                    }
                    ItemKind::Fn(func) => {
                        let kind = FnKind::Fn(FnCtxt::Free, visibility, &$($mut)? *func);
                        vis.visit_fn(kind, span, id)
                    }
                    ItemKind::Mod(safety, ident, mod_kind) => {
                        try_visit!(visit_safety(vis, safety));
                        try_visit!(vis.visit_ident(ident));
                        match mod_kind {
                            ModKind::Loaded(
                                items,
                                _inline,
                                ModSpans { inner_span, inject_use_span },
                                _,
                            ) => {
                                try_visit!(visit_items(vis, items));
                                try_visit!(visit_span(vis, inner_span));
                                try_visit!(visit_span(vis, inject_use_span));
                            }
                            ModKind::Unloaded => {}
                        }
                        V::Result::output()
                    }
                    ItemKind::ForeignMod(nm) => vis.visit_foreign_mod(nm),
                    ItemKind::GlobalAsm(asm) => vis.visit_inline_asm(asm),
                    ItemKind::TyAlias(box TyAlias {
                        defaultness,
                        ident,
                        generics,
                        $(${ignore($lt)} #[expect(unused)])?
                        where_clauses,
                        bounds,
                        ty,
                    }) => {
                        try_visit!(visit_defaultness(vis, defaultness));
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        try_visit!(visit_bounds(vis, bounds, BoundKind::Bound));
                        visit_opt!(vis, visit_ty, ty);
                        $(${ignore($mut)}
                            walk_ty_alias_where_clauses(vis, where_clauses);
                        )?
                        V::Result::output()
                    }
                    ItemKind::Enum(ident, generics, enum_definition) => {
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        visit_variants(vis, &$($mut)? enum_definition.variants)
                    }
                    ItemKind::Struct(ident, generics, variant_data)
                    | ItemKind::Union(ident, generics, variant_data) => {
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        vis.visit_variant_data(variant_data)
                    }
                    ItemKind::Impl(box Impl {
                        defaultness,
                        safety,
                        generics,
                        constness,
                        polarity,
                        of_trait,
                        self_ty,
                        items,
                    }) => {
                        try_visit!(visit_defaultness(vis, defaultness));
                        try_visit!(visit_safety(vis, safety));
                        try_visit!(vis.visit_generics(generics));
                        try_visit!(visit_constness(vis, constness));
                        try_visit!(visit_polarity(vis, polarity));
                        visit_opt!(vis, visit_trait_ref, of_trait);
                        try_visit!(vis.visit_ty(self_ty));
                        visit_assoc_items(vis, items, AssocCtxt::Impl { of_trait: of_trait.is_some() })
                    }
                    ItemKind::Trait(box Trait { safety, is_auto: _, ident, generics, bounds, items }) => {
                        try_visit!(visit_safety(vis, safety));
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        try_visit!(visit_bounds(vis, bounds, BoundKind::Bound));
                        visit_assoc_items(vis, items, AssocCtxt::Trait)
                    }
                    ItemKind::TraitAlias(ident, generics, bounds) => {
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        visit_bounds(vis, bounds, BoundKind::Bound)
                    }
                    ItemKind::MacCall(m) => vis.visit_mac_call(m),
                    ItemKind::MacroDef(ident, def) => {
                        try_visit!(vis.visit_ident(ident));
                        vis.visit_macro_def(def)
                    }
                    ItemKind::Delegation(box Delegation {
                        id,
                        qself,
                        path,
                        ident,
                        rename,
                        body,
                        from_glob: _,
                    }) => {
                        try_visit!(visit_id(vis, id));
                        try_visit!(vis.visit_qself(qself));
                        try_visit!(vis.visit_path(path));
                        try_visit!(vis.visit_ident(ident));
                        visit_opt!(vis, visit_ident, rename);
                        visit_opt!(vis, visit_block, body);
                        V::Result::output()
                    }
                    ItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                        try_visit!(vis.visit_qself(qself));
                        try_visit!(vis.visit_path(prefix));
                        if let Some(suffixes) = suffixes {
                            for (ident, rename) in suffixes {
                                try_visit!(vis.visit_ident(ident));
                                visit_opt!(vis, visit_ident, rename);
                            }
                        }
                        visit_opt!(vis, visit_block, body);
                        V::Result::output()
                    }
                }
            }
        }

        fn walk_const_item<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            item: &$($lt)? $($mut)? ConstItem,
        ) -> V::Result {
            let ConstItem { defaultness, ident, generics, ty, expr, define_opaque } = item;
            try_visit!(visit_defaultness(vis, defaultness));
            try_visit!(vis.visit_ident(ident));
            try_visit!(vis.visit_generics(generics));
            try_visit!(vis.visit_ty(ty));
            visit_opt!(vis, visit_expr, expr);
            walk_define_opaques(vis, define_opaque)
        }

        fn walk_foreign_mod<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, foreign_mod: &$($lt)? $($mut)? ForeignMod) -> V::Result {
            let ForeignMod { extern_span: _, safety, abi: _, items } = foreign_mod;
            try_visit!(visit_safety(vis, safety));
            visit_foreign_items(vis, items)
        }

        fn walk_define_opaques<$($lt,)? V: $Visitor$(<$lt>)?>(
            visitor: &mut V,
            define_opaque: &$($lt)? $($mut)? Option<ThinVec<(NodeId, Path)>>,
        ) -> V::Result {
            if let Some(define_opaque) = define_opaque {
                for (id, path) in define_opaque {
                    try_visit!(visit_id(visitor, id));
                    try_visit!(visitor.visit_path(path));
                }
            }
            V::Result::output()
        }

        impl WalkItemKind for AssocItemKind {
            type Ctxt = AssocCtxt;
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    AssocItemKind::Const(item) => {
                        walk_const_item(vis, item)
                    }
                    AssocItemKind::Fn(func) => {
                        vis.visit_fn(FnKind::Fn(FnCtxt::Assoc(ctxt), visibility, &$($mut)? *func), span, id)
                    }
                    AssocItemKind::Type(box TyAlias {
                        generics,
                        ident,
                        bounds,
                        ty,
                        defaultness,
                        $(${ignore($lt)} #[expect(unused)])?
                        where_clauses,
                    }) => {
                        try_visit!(visit_defaultness(vis, defaultness));
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        try_visit!(visit_bounds(vis, bounds, BoundKind::Bound));
                        visit_opt!(vis, visit_ty, ty);
                        $(${ignore($mut)}
                            walk_ty_alias_where_clauses(vis, where_clauses);
                        )?
                        V::Result::output()
                    }
                    AssocItemKind::MacCall(mac) => {
                        vis.visit_mac_call(mac)
                    }
                    AssocItemKind::Delegation(box Delegation {
                        id,
                        qself,
                        path,
                        ident,
                        rename,
                        body,
                        from_glob: _,
                    }) => {
                        try_visit!(visit_id(vis, id));
                        try_visit!(vis.visit_qself(qself));
                        try_visit!(vis.visit_path(path));
                        try_visit!(vis.visit_ident(ident));
                        visit_opt!(vis, visit_ident, rename);
                        visit_opt!(vis, visit_block, body);
                        V::Result::output()
                    }
                    AssocItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                        try_visit!(vis.visit_qself(qself));
                        try_visit!(vis.visit_path(prefix));
                        if let Some(suffixes) = suffixes {
                            for (ident, rename) in suffixes {
                                try_visit!(vis.visit_ident(ident));
                                visit_opt!(vis, visit_ident, rename);
                            }
                        }
                        visit_opt!(vis, visit_block, body);
                        V::Result::output()
                    }
                }
            }
        }

        impl WalkItemKind for ForeignItemKind {
            type Ctxt = ();
            fn walk<$($lt,)? V: $Visitor$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                span: Span,
                id: NodeId,
                visibility: &$($lt)? $($mut)? Visibility,
                _ctxt: Self::Ctxt,
                vis: &mut V,
            ) -> V::Result {
                match self {
                    ForeignItemKind::Static(box StaticItem {
                        ident,
                        ty,
                        mutability: _,
                        expr,
                        safety: _,
                        define_opaque,
                    }) => {
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_ty(ty));
                        visit_opt!(vis, visit_expr, expr);
                        walk_define_opaques(vis, define_opaque)
                    }
                    ForeignItemKind::Fn(func) => {
                        vis.visit_fn(FnKind::Fn(FnCtxt::Foreign, visibility, &$($mut)?*func), span, id)
                    }
                    ForeignItemKind::TyAlias(box TyAlias {
                        defaultness,
                        ident,
                        generics,
                        bounds,
                        ty,
                        $(${ignore($lt)} #[expect(unused)])?
                        where_clauses,
                    }) => {
                        try_visit!(visit_defaultness(vis, defaultness));
                        try_visit!(vis.visit_ident(ident));
                        try_visit!(vis.visit_generics(generics));
                        try_visit!(visit_bounds(vis, bounds, BoundKind::Bound));
                        visit_opt!(vis, visit_ty, ty);
                        $(${ignore($mut)}
                            walk_ty_alias_where_clauses(vis, where_clauses);
                        )?
                        V::Result::output()
                    }
                    ForeignItemKind::MacCall(mac) => {
                        vis.visit_mac_call(mac)
                    }
                }
            }
        }

        fn walk_coroutine_kind<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            coroutine_kind: &$($lt)? $($mut)? CoroutineKind,
        ) -> V::Result {
            let (CoroutineKind::Async { span, closure_id, return_impl_trait_id }
                | CoroutineKind::Gen { span, closure_id, return_impl_trait_id }
                | CoroutineKind::AsyncGen { span, closure_id, return_impl_trait_id })
                = coroutine_kind;
            try_visit!(visit_id(vis, closure_id));
            try_visit!(visit_id(vis, return_impl_trait_id));
            visit_span(vis, span)
        }

        pub fn walk_pat<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            pattern: &$($lt)? $($mut)? Pat
        ) -> V::Result {
            let Pat { id, kind, span, tokens: _ } = pattern;
            try_visit!(visit_id(vis, id));
            match kind {
                PatKind::Err(_guar) => {}
                PatKind::Missing | PatKind::Wild | PatKind::Rest | PatKind::Never => {}
                PatKind::Ident(_bmode, ident, optional_subpattern) => {
                    try_visit!(vis.visit_ident(ident));
                    visit_opt!(vis, visit_pat, optional_subpattern);
                }
                PatKind::Expr(expression) => try_visit!(vis.visit_expr(expression)),
                PatKind::TupleStruct(opt_qself, path, elems) => {
                    try_visit!(vis.visit_qself(opt_qself));
                    try_visit!(vis.visit_path(path));
                    walk_list!(vis, visit_pat, elems);
                }
                PatKind::Path(opt_qself, path) => {
                    try_visit!(vis.visit_qself(opt_qself));
                    try_visit!(vis.visit_path(path))
                }
                PatKind::Struct(opt_qself, path, fields, _rest) => {
                    try_visit!(vis.visit_qself(opt_qself));
                    try_visit!(vis.visit_path(path));
                    try_visit!(visit_pat_fields(vis, fields));
                }
                PatKind::Box(subpattern) | PatKind::Deref(subpattern) | PatKind::Paren(subpattern) => {
                    try_visit!(vis.visit_pat(subpattern));
                }
                PatKind::Ref(subpattern, _ /*mutbl*/) => {
                    try_visit!(vis.visit_pat(subpattern));
                }
                PatKind::Range(lower_bound, upper_bound, _end) => {
                    visit_opt!(vis, visit_expr, lower_bound);
                    visit_opt!(vis, visit_expr, upper_bound);
                    try_visit!(visit_span(vis, span));
                }
                PatKind::Guard(subpattern, guard_condition) => {
                    try_visit!(vis.visit_pat(subpattern));
                    try_visit!(vis.visit_expr(guard_condition));
                }
                PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
                    walk_list!(vis, visit_pat, elems);
                }
                PatKind::MacCall(mac) => try_visit!(vis.visit_mac_call(mac)),
            }
            visit_span(vis, span)
        }

        pub fn walk_anon_const<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            constant: &$($lt)? $($mut)? AnonConst,
        ) -> V::Result {
            let AnonConst { id, value } = constant;
            try_visit!(visit_id(vis, id));
            vis.visit_expr(value)
        }

        pub fn walk_path_segment<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            segment: &$($lt)? $($mut)? PathSegment,
        ) -> V::Result {
            let PathSegment { ident, id, args } = segment;
            try_visit!(visit_id(vis, id));
            try_visit!(vis.visit_ident(ident));
            visit_opt!(vis, visit_generic_args, args);
            V::Result::output()
        }

        pub fn walk_block<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            block: &$($lt)? $($mut)? Block
        ) -> V::Result {
            let Block { stmts, id, rules: _, span, tokens: _ } = block;
            try_visit!(visit_id(vis, id));
            try_visit!(visit_stmts(vis, stmts));
            visit_span(vis, span)
        }


        pub fn walk_ty<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V, ty: &$($lt)? $($mut)? Ty
        ) -> V::Result {
            let Ty { id, kind, span, tokens: _ } = ty;
            try_visit!(visit_id(vis, id));
            match kind {
                TyKind::Err(_guar) => {}
                TyKind::Infer | TyKind::ImplicitSelf | TyKind::Dummy | TyKind::Never | TyKind::CVarArgs => {}
                TyKind::Slice(ty) | TyKind::Paren(ty) => try_visit!(vis.visit_ty(ty)),
                TyKind::Ptr(MutTy { ty, mutbl: _ }) => try_visit!(vis.visit_ty(ty)),
                TyKind::Ref(opt_lifetime, MutTy { ty, mutbl: _ })
                | TyKind::PinnedRef(opt_lifetime, MutTy { ty, mutbl: _ }) => {
                    // FIXME(fee1-dead) asymmetry
                    visit_opt!(vis, visit_lifetime, opt_lifetime$(${ignore($lt)}, LifetimeCtxt::Ref)?);
                    try_visit!(vis.visit_ty(ty));
                }
                TyKind::Tup(tuple_element_types) => {
                    walk_list!(vis, visit_ty, tuple_element_types);
                }
                TyKind::BareFn(function_declaration) => {
                    let BareFnTy { safety, ext: _, generic_params, decl, decl_span } =
                        &$($mut)? **function_declaration;
                    try_visit!(visit_safety(vis, safety));
                    try_visit!(visit_generic_params(vis, generic_params));
                    try_visit!(vis.visit_fn_decl(decl));
                    try_visit!(visit_span(vis, decl_span));
                }
                TyKind::UnsafeBinder(binder) => {
                    try_visit!(visit_generic_params(vis, &$($mut)? binder.generic_params));
                    try_visit!(vis.visit_ty(&$($mut)? binder.inner_ty));
                }
                TyKind::Path(maybe_qself, path) => {
                    try_visit!(vis.visit_qself(maybe_qself));
                    try_visit!(vis.visit_path(path));
                }
                TyKind::Pat(ty, pat) => {
                    try_visit!(vis.visit_ty(ty));
                    try_visit!(vis.visit_ty_pat(pat));
                }
                TyKind::Array(ty, length) => {
                    try_visit!(vis.visit_ty(ty));
                    try_visit!(vis.visit_anon_const(length));
                }
                TyKind::TraitObject(bounds, _syntax) => {
                    walk_list!(vis, visit_param_bound, bounds, BoundKind::TraitObject);
                }
                TyKind::ImplTrait(id, bounds) => {
                    try_visit!(visit_id(vis, id));
                    walk_list!(vis, visit_param_bound, bounds, BoundKind::Impl);
                }
                TyKind::Typeof(expression) => try_visit!(vis.visit_anon_const(expression)),

                TyKind::MacCall(mac) => try_visit!(vis.visit_mac_call(mac)),
            }
            visit_span(vis, span)
        }

        pub fn walk_crate<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            krate: &$($lt)? $($mut)? Crate,
        ) -> V::Result {
            let Crate { attrs, items, spans, id, is_placeholder: _ } = krate;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(visit_items(vis, items));
            let ModSpans { inner_span, inject_use_span } = spans;
            try_visit!(visit_span(vis, inner_span));
            visit_span(vis, inject_use_span)
        }

        pub fn walk_local<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            local: &$($lt)? $($mut)? Local,
        ) -> V::Result {
            let Local { id, super_, pat, ty, kind, span, colon_sp, attrs, tokens: _ } = local;
            if let Some(sp) = super_ {
                try_visit!(visit_span(vis, sp));
            }
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_pat(pat));
            visit_opt!(vis, visit_ty, ty);
            match kind {
                LocalKind::Decl => {}
                LocalKind::Init(init) => {
                    try_visit!(vis.visit_expr(init))
                }
                LocalKind::InitElse(init, els) => {
                    try_visit!(vis.visit_expr(init));
                    try_visit!(vis.visit_block(els));
                }
            }
            if let Some(sp) = colon_sp {
                try_visit!(visit_span(vis, sp));
            }
            visit_span(vis, span)
        }

        pub fn walk_poly_trait_ref<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            p: &$($lt)? $($mut)? PolyTraitRef,
        ) -> V::Result {
            let PolyTraitRef { bound_generic_params, modifiers, trait_ref, span } = p;
            try_visit!(visit_modifiers(vis, modifiers));
            try_visit!(visit_generic_params(vis, bound_generic_params));
            try_visit!(vis.visit_trait_ref(trait_ref));
            visit_span(vis, span)
        }

        pub fn walk_trait_ref<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            TraitRef { path, ref_id }: &$($lt)? $($mut)? TraitRef,
        ) -> V::Result {
            try_visit!(vis.visit_path(path));
            visit_id(vis, ref_id)
        }

        pub fn walk_variant<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            variant: &$($lt)? $($mut)? Variant,
        ) -> V::Result {
            let Variant { attrs, id, span, vis: visibility, ident, data, disr_expr, is_placeholder: _ } = variant;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_vis(visibility));
            try_visit!(vis.visit_ident(ident));
            try_visit!(vis.visit_variant_data(data));
            $(${ignore($lt)} visit_opt!(vis, visit_variant_discr, disr_expr); )?
            $(${ignore($mut)} visit_opt!(vis, visit_anon_const, disr_expr); )?
            visit_span(vis, span)
        }

        pub fn walk_expr_field<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            f: &$($lt)? $($mut)? ExprField,
        ) -> V::Result {
            let ExprField { attrs, id, span, ident, expr, is_shorthand: _, is_placeholder: _ } = f;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_ident(ident));
            try_visit!(vis.visit_expr(expr));
            visit_span(vis, span)
        }

        pub fn walk_pat_field<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            fp: &$($lt)? $($mut)? PatField,
        ) -> V::Result {
            let PatField { ident, pat, is_shorthand: _, attrs, id, span, is_placeholder: _ } = fp;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_ident(ident));
            try_visit!(vis.visit_pat(pat));
            visit_span(vis, span)
        }

        pub fn walk_ty_pat<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            tp: &$($lt)? $($mut)? TyPat,
        ) -> V::Result {
            let TyPat { id, kind, span, tokens: _ } = tp;
            try_visit!(visit_id(vis, id));
            match kind {
                TyPatKind::Range(start, end, _include_end) => {
                    visit_opt!(vis, visit_anon_const, start);
                    visit_opt!(vis, visit_anon_const, end);
                }
                TyPatKind::Or(variants) => walk_list!(vis, visit_ty_pat, variants),
                TyPatKind::Err(_) => {}
            }
            visit_span(vis, span)
        }

        fn walk_qself<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            qself: &$($lt)? $($mut)? Option<P<QSelf>>,
        ) -> V::Result {
            if let Some(qself) = qself {
                let QSelf { ty, path_span, position: _ } = &$($mut)? **qself;
                try_visit!(vis.visit_ty(ty));
                try_visit!(visit_span(vis, path_span));
            }
            V::Result::output()
        }

        pub fn walk_path<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            path: &$($lt)? $($mut)? Path,
        ) -> V::Result {
            let Path { span, segments, tokens: _ } = path;
            walk_list!(vis, visit_path_segment, segments);
            visit_span(vis, span)
        }

        pub fn walk_use_tree<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            use_tree: &$($lt)? $($mut)? UseTree,
        ) -> V::Result {
            let UseTree { prefix, kind, span } = use_tree;
            try_visit!(vis.visit_path(prefix));
            match kind {
                UseTreeKind::Simple(rename) => {
                    // The extra IDs are handled during AST lowering.
                    visit_opt!(vis, visit_ident, rename);
                }
                UseTreeKind::Glob => {}
                UseTreeKind::Nested { items, span } => {
                    for (nested_tree, nested_id) in items {
                        try_visit!(visit_nested_use_tree(vis, nested_tree, nested_id));
                    }
                    try_visit!(visit_span(vis, span));
                }
            }
            visit_span(vis, span)
        }

        pub fn walk_generic_args<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            generic_args: &$($lt)? $($mut)? GenericArgs
        ) -> V::Result {
            match generic_args {
                GenericArgs::AngleBracketed(AngleBracketedArgs { span, args }) => {
                    for arg in args {
                        match arg {
                            AngleBracketedArg::Arg(a) => try_visit!(vis.visit_generic_arg(a)),
                            AngleBracketedArg::Constraint(c) => {
                                try_visit!(vis.visit_assoc_item_constraint(c))
                            }
                        }
                    }
                    visit_span(vis, span)
                }
                GenericArgs::Parenthesized(data) => {
                    let ParenthesizedArgs { span, inputs, inputs_span, output } = data;
                    walk_list!(vis, visit_ty, inputs);
                    try_visit!(vis.visit_fn_ret_ty(output));
                    try_visit!(visit_span(vis, span));
                    visit_span(vis, inputs_span)
                }
                GenericArgs::ParenthesizedElided(span) => visit_span(vis, span)
            }
        }

        pub fn walk_generic_arg<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            generic_arg: &$($lt)? $($mut)? GenericArg,
        ) -> V::Result {
            match generic_arg {
                GenericArg::Lifetime(lt) => vis.visit_lifetime(lt, $(${ignore($lt)} LifetimeCtxt::GenericArg)? ),
                GenericArg::Type(ty) => vis.visit_ty(ty),
                GenericArg::Const(ct) => vis.visit_anon_const(ct),
            }
        }

        pub fn walk_assoc_item_constraint<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            constraint: &$($lt)? $($mut)? AssocItemConstraint,
        ) -> V::Result {
            let AssocItemConstraint { id, ident, gen_args, kind, span } = constraint;
            try_visit!(visit_id(vis, id));
            try_visit!(vis.visit_ident(ident));
            visit_opt!(vis, visit_generic_args, gen_args);
            match kind {
                AssocItemConstraintKind::Equality { term } => match term {
                    Term::Ty(ty) => try_visit!(vis.visit_ty(ty)),
                    Term::Const(c) => try_visit!(vis.visit_anon_const(c)),
                },
                AssocItemConstraintKind::Bound { bounds } => {
                    try_visit!(visit_bounds(vis, bounds, BoundKind::Bound));
                }
            }
            visit_span(vis, span)
        }

        pub fn walk_param_bound<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, bound: &$($lt)? $($mut)? GenericBound) -> V::Result {
            match bound {
                GenericBound::Trait(trait_ref) => vis.visit_poly_trait_ref(trait_ref),
                GenericBound::Outlives(lifetime) => vis.visit_lifetime(lifetime, $(${ignore($lt)} LifetimeCtxt::Bound)?),
                GenericBound::Use(args, span) => {
                    walk_list!(vis, visit_precise_capturing_arg, args);
                    visit_span(vis, span)
                }
            }
        }

        pub fn walk_precise_capturing_arg<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            arg: &$($lt)? $($mut)? PreciseCapturingArg,
        ) -> V::Result {
            match arg {
                PreciseCapturingArg::Lifetime(lt) => vis.visit_lifetime(lt, $(${ignore($lt)} LifetimeCtxt::GenericArg)?),
                PreciseCapturingArg::Arg(path, id) => {
                    try_visit!(visit_id(vis, id));
                    vis.visit_path(path)
                }
            }
        }

        pub fn walk_generic_param<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            param: &$($lt)? $($mut)? GenericParam,
        ) -> V::Result {
            let GenericParam { id, ident, attrs, bounds, is_placeholder: _, kind, colon_span } =
                param;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_ident(ident));
            walk_list!(vis, visit_param_bound, bounds, BoundKind::Bound);
            match kind {
                GenericParamKind::Lifetime => (),
                GenericParamKind::Type { default } => visit_opt!(vis, visit_ty, default),
                GenericParamKind::Const { ty, default, kw_span: _ } => {
                    try_visit!(vis.visit_ty(ty));
                    visit_opt!(vis, visit_anon_const, default);
                }
            }
            if let Some(sp) = colon_span {
                try_visit!(visit_span(vis, sp))
            }
            V::Result::output()
        }

        pub fn walk_generics<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, generics: &$($lt)? $($mut)? Generics) -> V::Result {
            let Generics { params, where_clause, span } = generics;
            let WhereClause { has_where_token: _, predicates, span: where_clause_span } = where_clause;
            try_visit!(visit_generic_params(vis, params));
            try_visit!(visit_where_predicates(vis, predicates));
            try_visit!(visit_span(vis, span));
            visit_span(vis, where_clause_span)
        }

        pub fn walk_contract<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, c: &$($lt)? $($mut)? FnContract) -> V::Result {
            let FnContract { requires, ensures } = c;
            visit_opt!(vis, visit_expr, requires);
            visit_opt!(vis, visit_expr, ensures);
            V::Result::output()
        }

        pub fn walk_where_predicate<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            predicate: &$($lt)? $($mut)? WherePredicate,
        ) -> V::Result {
            let WherePredicate { attrs, kind, id, span, is_placeholder: _ } = predicate;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(visit_span(vis, span));
            vis.visit_where_predicate_kind(kind)
        }

        pub fn walk_closure_binder<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            binder: &$($lt)? $($mut)? ClosureBinder,
        ) -> V::Result {
            match binder {
                ClosureBinder::NotPresent => {}
                ClosureBinder::For { generic_params, span } => {
                    try_visit!(visit_generic_params(vis, generic_params));
                    try_visit!(visit_span(vis, span));
                }
            }
            V::Result::output()
        }

        pub fn walk_where_predicate_kind<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            kind: &$($lt)? $($mut)? WherePredicateKind,
        ) -> V::Result {
            match kind {
                WherePredicateKind::BoundPredicate(WhereBoundPredicate {
                    bounded_ty,
                    bounds,
                    bound_generic_params,
                }) => {
                    try_visit!(visit_generic_params(vis, bound_generic_params));
                    try_visit!(vis.visit_ty(bounded_ty));
                    walk_list!(vis, visit_param_bound, bounds, BoundKind::Bound);
                }
                WherePredicateKind::RegionPredicate(WhereRegionPredicate { lifetime, bounds }) => {
                    try_visit!(vis.visit_lifetime(lifetime, $(${ignore($lt)} LifetimeCtxt::Bound )?));
                    walk_list!(vis, visit_param_bound, bounds, BoundKind::Bound);
                }
                WherePredicateKind::EqPredicate(WhereEqPredicate { lhs_ty, rhs_ty }) => {
                    try_visit!(vis.visit_ty(lhs_ty));
                    try_visit!(vis.visit_ty(rhs_ty));
                }
            }
            V::Result::output()
        }

        pub fn walk_fn_decl<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            FnDecl { inputs, output }: &$($lt)? $($mut)? FnDecl,
        ) -> V::Result {
            try_visit!(visit_params(vis, inputs));
            vis.visit_fn_ret_ty(output)
        }

        pub fn walk_fn_ret_ty<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, ret_ty: &$($lt)? $($mut)? FnRetTy) -> V::Result {
            match ret_ty {
                FnRetTy::Default(span) => visit_span(vis, span),
                FnRetTy::Ty(output_ty) => vis.visit_ty(output_ty),
            }
        }

        pub fn walk_fn<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, kind: FnKind<$($lt)? $(${ignore($mut)} '_)?>) -> V::Result {
            match kind {
                FnKind::Fn(
                    _ctxt,
                    _vis,
                    Fn {
                        defaultness,
                        ident,
                        sig: FnSig { header, decl, span },
                        generics,
                        contract,
                        body,
                        define_opaque,
                    },
                ) => {
                    // Visibility is visited as a part of the item.
                    try_visit!(visit_defaultness(vis, defaultness));
                    try_visit!(vis.visit_ident(ident));
                    try_visit!(vis.visit_fn_header(header));
                    try_visit!(vis.visit_generics(generics));
                    try_visit!(vis.visit_fn_decl(decl));
                    visit_opt!(vis, visit_contract, contract);
                    visit_opt!(vis, visit_block, body);
                    try_visit!(visit_span(vis, span));
                    walk_define_opaques(vis, define_opaque)
                }
                FnKind::Closure(binder, coroutine_kind, decl, body) => {
                    try_visit!(vis.visit_closure_binder(binder));
                    visit_opt!(vis, visit_coroutine_kind, coroutine_kind);
                    try_visit!(vis.visit_fn_decl(decl));
                    vis.visit_expr(body)
                }
            }
        }

        pub fn walk_variant_data<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, data: &$($lt)? $($mut)? VariantData) -> V::Result {
            match data {
                VariantData::Struct { fields, recovered: _ } => {
                    visit_field_defs(vis, fields)
                }
                VariantData::Tuple(fields, id) => {
                    try_visit!(visit_id(vis, id));
                    visit_field_defs(vis, fields)
                }
                VariantData::Unit(id) => visit_id(vis, id),
            }
        }

        pub fn walk_field_def<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, field: &$($lt)? $($mut)? FieldDef) -> V::Result {
            let FieldDef { attrs, id, span, vis: visibility, ident, ty, is_placeholder: _, safety: _, default } =
                field;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_vis(visibility));
            visit_opt!(vis, visit_ident, ident);
            try_visit!(vis.visit_ty(ty));
            visit_opt!(vis, visit_anon_const, default);
            visit_span(vis, span)
        }

        fn visit_delim_args<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, args: &$($lt)? $($mut)? DelimArgs) -> V::Result {
            let DelimArgs { dspan, delim: _, tokens: _ } = args;
            let DelimSpan { open, close } = dspan;
            try_visit!(visit_span(vis, open));
            visit_span(vis, close)
        }

        pub fn walk_mac<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, mac: &$($lt)? $($mut)? MacCall) -> V::Result {
            let MacCall { path, args } = mac;
            try_visit!(vis.visit_path(path));
            visit_delim_args(vis, args)
        }

        fn walk_macro_def<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, macro_def: &$($lt)? $($mut)? MacroDef) -> V::Result {
            let MacroDef { body, macro_rules: _ } = macro_def;
            visit_delim_args(vis, body)
        }

        pub fn walk_inline_asm<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, asm: &$($lt)? $($mut)? InlineAsm) -> V::Result {
            // FIXME: Visit spans inside all this currently ignored stuff.
            let InlineAsm {
                asm_macro: _,
                template: _,
                template_strs: _,
                operands,
                clobber_abis: _,
                options: _,
                line_spans: _,
            } = asm;
            for (op, span) in operands {
                match op {
                    InlineAsmOperand::In { expr, reg: _ }
                    | InlineAsmOperand::Out { expr: Some(expr), reg: _, late: _ }
                    | InlineAsmOperand::InOut { expr, reg: _, late: _ } => {
                        try_visit!(vis.visit_expr(expr))
                    }
                    InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
                    InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                        try_visit!(vis.visit_expr(in_expr));
                        visit_opt!(vis, visit_expr, out_expr);
                    }
                    InlineAsmOperand::Const { anon_const } => {
                        try_visit!(vis.visit_anon_const(anon_const))
                    }
                    InlineAsmOperand::Sym { sym } => try_visit!(vis.visit_inline_asm_sym(sym)),
                    InlineAsmOperand::Label { block } => try_visit!(vis.visit_block(block)),
                }
                try_visit!(visit_span(vis, span));
            }
            V::Result::output()
        }

        pub fn walk_inline_asm_sym<$($lt,)? V: $Visitor$(<$lt>)?>(
            vis: &mut V,
            InlineAsmSym { id, qself, path }: &$($lt)? $($mut)? InlineAsmSym,
        ) -> V::Result {
            try_visit!(visit_id(vis, id));
            try_visit!(vis.visit_qself(qself));
            vis.visit_path(path)
        }

        // FIXME: visit the template exhaustively.
        pub fn walk_format_args<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, fmt: &$($lt)? $($mut)? FormatArgs) -> V::Result {
            let FormatArgs { span, template: _, arguments, uncooked_fmt_str: _ } = fmt;
            let args = $(${ignore($mut)} arguments.all_args_mut())? $(${ignore($lt)} arguments.all_args())? ;
            for FormatArgument { kind, expr } in args {
                match kind {
                    FormatArgumentKind::Named(ident) | FormatArgumentKind::Captured(ident) => {
                        try_visit!(vis.visit_ident(ident))
                    }
                    FormatArgumentKind::Normal => {}
                }
                try_visit!(vis.visit_expr(expr));
            }
            visit_span(vis, span)
        }

        pub fn walk_expr<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, expression: &$($lt)? $($mut)? Expr) -> V::Result {
            let Expr { id, kind, span, attrs, tokens: _ } = expression;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            match kind {
                ExprKind::Array(exprs) => {
                    try_visit!(visit_exprs(vis, exprs));
                }
                ExprKind::ConstBlock(anon_const) => try_visit!(vis.visit_anon_const(anon_const)),
                ExprKind::Repeat(element, count) => {
                    try_visit!(vis.visit_expr(element));
                    try_visit!(vis.visit_anon_const(count));
                }
                ExprKind::Struct(se) => {
                    let StructExpr { qself, path, fields, rest } = &$($mut)?**se;
                    try_visit!(vis.visit_qself(qself));
                    try_visit!(vis.visit_path(path));
                    try_visit!(visit_expr_fields(vis, fields));
                    match rest {
                        StructRest::Base(expr) => try_visit!(vis.visit_expr(expr)),
                        StructRest::Rest(_span) => {}
                        StructRest::None => {}
                    }
                }
                ExprKind::Tup(exprs) => {
                    try_visit!(visit_exprs(vis, exprs));
                }
                ExprKind::Call(callee_expression, arguments) => {
                    try_visit!(vis.visit_expr(callee_expression));
                    try_visit!(visit_exprs(vis, arguments));
                }
                ExprKind::MethodCall(box MethodCall { seg, receiver, args, span }) => {
                    try_visit!(vis.visit_method_receiver_expr(receiver));
                    try_visit!(vis.visit_path_segment(seg));
                    try_visit!(visit_exprs(vis, args));
                    try_visit!(visit_span(vis, span));
                }
                ExprKind::Binary(Spanned { span, node: _ }, left_expression, right_expression) => {
                    try_visit!(vis.visit_expr(left_expression));
                    try_visit!(vis.visit_expr(right_expression));
                    try_visit!(visit_span(vis, span))
                }
                ExprKind::AddrOf(_kind, _mutbl, subexpression) => {
                    try_visit!(vis.visit_expr(subexpression));
                }
                ExprKind::Unary(_op, subexpression) => {
                    try_visit!(vis.visit_expr(subexpression));
                }
                ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) => {
                    try_visit!(vis.visit_expr(subexpression));
                    try_visit!(vis.visit_ty(typ));
                }
                ExprKind::Let(pat, expr, span, _recovered) => {
                    try_visit!(vis.visit_pat(pat));
                    try_visit!(vis.visit_expr(expr));
                    try_visit!(visit_span(vis, span))
                }
                ExprKind::If(head_expression, if_block, optional_else) => {
                    try_visit!(vis.visit_expr(head_expression));
                    try_visit!(vis.visit_block(if_block));
                    visit_opt!(vis, visit_expr, optional_else);
                }
                ExprKind::While(subexpression, block, opt_label) => {
                    visit_opt!(vis, visit_label, opt_label);
                    try_visit!(vis.visit_expr(subexpression));
                    try_visit!(vis.visit_block(block));
                }
                ExprKind::ForLoop { pat, iter, body, label, kind: _ } => {
                    visit_opt!(vis, visit_label, label);
                    try_visit!(vis.visit_pat(pat));
                    try_visit!(vis.visit_expr(iter));
                    try_visit!(vis.visit_block(body));
                }
                ExprKind::Loop(block, opt_label, span) => {
                    visit_opt!(vis, visit_label, opt_label);
                    try_visit!(vis.visit_block(block));
                    try_visit!(visit_span(vis, span))
                }
                ExprKind::Match(subexpression, arms, _kind) => {
                    try_visit!(vis.visit_expr(subexpression));
                    try_visit!(visit_arms(vis, arms));
                }
                ExprKind::Closure(box Closure {
                    binder,
                    capture_clause,
                    coroutine_kind,
                    constness,
                    movability: _,
                    fn_decl,
                    body,
                    fn_decl_span,
                    fn_arg_span,
                }) => {
                    try_visit!(visit_constness(vis, constness));
                    try_visit!(vis.visit_capture_by(capture_clause));
                    try_visit!(vis.visit_fn(
                        FnKind::Closure(binder, coroutine_kind, fn_decl, body),
                        *span,
                        *id
                    ));
                    try_visit!(visit_span(vis, fn_decl_span));
                    try_visit!(visit_span(vis, fn_arg_span));
                }
                ExprKind::Block(block, opt_label) => {
                    visit_opt!(vis, visit_label, opt_label);
                    try_visit!(vis.visit_block(block));
                }
                ExprKind::Gen(_capt, body, _kind, decl_span) => {
                    try_visit!(vis.visit_block(body));
                    try_visit!(visit_span(vis, decl_span));
                }
                ExprKind::Await(expr, span) => {
                    try_visit!(vis.visit_expr(expr));
                    try_visit!(visit_span(vis, span));
                }
                ExprKind::Use(expr, span) => {
                    try_visit!(vis.visit_expr(expr));
                    try_visit!(visit_span(vis, span));
                }
                ExprKind::Assign(lhs, rhs, span) => {
                    try_visit!(vis.visit_expr(lhs));
                    try_visit!(vis.visit_expr(rhs));
                    try_visit!(visit_span(vis, span));
                }
                ExprKind::AssignOp(_op, left_expression, right_expression) => {
                    try_visit!(vis.visit_expr(left_expression));
                    try_visit!(vis.visit_expr(right_expression));
                }
                ExprKind::Field(subexpression, ident) => {
                    try_visit!(vis.visit_expr(subexpression));
                    try_visit!(vis.visit_ident(ident));
                }
                ExprKind::Index(main_expression, index_expression, span) => {
                    try_visit!(vis.visit_expr(main_expression));
                    try_visit!(vis.visit_expr(index_expression));
                    try_visit!(visit_span(vis, span));
                }
                ExprKind::Range(start, end, _limit) => {
                    visit_opt!(vis, visit_expr, start);
                    visit_opt!(vis, visit_expr, end);
                }
                ExprKind::Underscore => {}
                ExprKind::Path(maybe_qself, path) => {
                    try_visit!(vis.visit_qself(maybe_qself));
                    try_visit!(vis.visit_path(path));
                }
                ExprKind::Break(opt_label, opt_expr) => {
                    visit_opt!(vis, visit_label, opt_label);
                    visit_opt!(vis, visit_expr, opt_expr);
                }
                ExprKind::Continue(opt_label) => {
                    visit_opt!(vis, visit_label, opt_label);
                }
                ExprKind::Ret(optional_expression) => {
                    visit_opt!(vis, visit_expr, optional_expression);
                }
                ExprKind::Yeet(optional_expression) => {
                    visit_opt!(vis, visit_expr, optional_expression);
                }
                ExprKind::Become(expr) => try_visit!(vis.visit_expr(expr)),
                ExprKind::MacCall(mac) => try_visit!(vis.visit_mac_call(mac)),
                ExprKind::Paren(subexpression) => try_visit!(vis.visit_expr(subexpression)),
                ExprKind::InlineAsm(asm) => try_visit!(vis.visit_inline_asm(asm)),
                ExprKind::FormatArgs(f) => try_visit!(vis.visit_format_args(f)),
                ExprKind::OffsetOf(container, fields) => {
                    try_visit!(vis.visit_ty(container));
                    walk_list!(vis, visit_ident, fields);
                }
                ExprKind::Yield(kind) => {
                    match kind {
                        YieldKind::Postfix(expr) => {
                            try_visit!(vis.visit_expr(expr));
                        }
                        YieldKind::Prefix(expr) => {
                            visit_opt!(vis, visit_expr, expr);
                        }
                    }
                }
                ExprKind::Try(subexpression) => try_visit!(vis.visit_expr(subexpression)),
                ExprKind::TryBlock(body) => try_visit!(vis.visit_block(body)),
                ExprKind::Lit(_token) => {}
                ExprKind::IncludedBytes(_bytes) => {}
                ExprKind::UnsafeBinderCast(_kind, expr, ty) => {
                    try_visit!(vis.visit_expr(expr));
                    visit_opt!(vis, visit_ty, ty);
                }
                ExprKind::Err(_guar) => {}
                ExprKind::Dummy => {}
            }

            visit_span(vis, span)
        }

        pub fn walk_param<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, param: &$($lt)? $($mut)? Param) -> V::Result {
            let Param { attrs, ty, pat, id, span, is_placeholder: _ } = param;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_pat(pat));
            try_visit!(vis.visit_ty(ty));
            visit_span(vis, span)
        }

        pub fn walk_arm<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, arm: &$($lt)? $($mut)? Arm) -> V::Result {
            let Arm { attrs, pat, guard, body, span, id, is_placeholder: _ } = arm;
            try_visit!(visit_id(vis, id));
            walk_list!(vis, visit_attribute, attrs);
            try_visit!(vis.visit_pat(pat));
            visit_opt!(vis, visit_expr, guard);
            visit_opt!(vis, visit_expr, body);
            visit_span(vis, span)
        }

        pub fn walk_vis<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, visibility: &$($lt)? $($mut)? Visibility) -> V::Result {
            let Visibility { kind, span, tokens: _ } = visibility;
            match kind {
                VisibilityKind::Restricted { path, id, shorthand: _ } => {
                    try_visit!(visit_id(vis, id));
                    try_visit!(vis.visit_path(path));
                }
                VisibilityKind::Public | VisibilityKind::Inherited => {}
            }
            visit_span(vis, span)
        }

        pub fn walk_attribute<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, attr: &$($lt)? $($mut)? Attribute) -> V::Result {
            let Attribute { kind, id: _, style: _, span } = attr;
            match kind {
                AttrKind::Normal(normal) => {
                    let NormalAttr { item, tokens: _ } = &$($mut)?**normal;
                    let AttrItem { unsafety: _, path, args, tokens: _ } = item;
                    try_visit!(vis.visit_path(path));
                    try_visit!(walk_attr_args(vis, args));
                }
                AttrKind::DocComment(_kind, _sym) => {}
            }
            visit_span(vis, span)
        }

        pub fn walk_attr_args<$($lt,)? V: $Visitor$(<$lt>)?>(vis: &mut V, args: &$($lt)? $($mut)? AttrArgs) -> V::Result {
            match args {
                AttrArgs::Empty => {}
                AttrArgs::Delimited(args) => try_visit!(visit_delim_args(vis, args)),
                AttrArgs::Eq { eq_span, expr } => {
                    try_visit!(vis.visit_expr(expr));
                    try_visit!(visit_span(vis, eq_span));
                }
            }
            V::Result::output()
        }
    };
}

common_visitor_and_walkers!(Visitor<'a>);

macro_rules! generate_list_visit_fns {
    ($($name:ident, $Ty:ty, $visit_fn:ident$(, $param:ident: $ParamTy:ty)*;)+) => {
        $(
            fn $name<'a, V: Visitor<'a>>(
                vis: &mut V,
                values: &'a ThinVec<$Ty>,
                $(
                    $param: $ParamTy,
                )*
            ) -> V::Result {
                walk_list!(vis, $visit_fn, values$(,$param)*);
                V::Result::output()
            }
        )+
    }
}

generate_list_visit_fns! {
    visit_items, P<Item>, visit_item;
    visit_foreign_items, P<ForeignItem>, visit_foreign_item;
    visit_generic_params, GenericParam, visit_generic_param;
    visit_stmts, Stmt, visit_stmt;
    visit_exprs, P<Expr>, visit_expr;
    visit_expr_fields, ExprField, visit_expr_field;
    visit_pat_fields, PatField, visit_pat_field;
    visit_variants, Variant, visit_variant;
    visit_assoc_items, P<AssocItem>, visit_assoc_item, ctxt: AssocCtxt;
    visit_where_predicates, WherePredicate, visit_where_predicate;
    visit_params, Param, visit_param;
    visit_field_defs, FieldDef, visit_field_def;
    visit_arms, Arm, visit_arm;
}

#[expect(rustc::pass_by_value)] // needed for symmetry with mut_visit
fn visit_nested_use_tree<'a, V: Visitor<'a>>(
    vis: &mut V,
    nested_tree: &'a UseTree,
    &nested_id: &NodeId,
) -> V::Result {
    vis.visit_nested_use_tree(nested_tree, nested_id)
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) -> V::Result {
    let Stmt { id, kind, span: _ } = statement;
    try_visit!(visit_id(visitor, id));
    match kind {
        StmtKind::Let(local) => try_visit!(visitor.visit_local(local)),
        StmtKind::Item(item) => try_visit!(visitor.visit_item(item)),
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => try_visit!(visitor.visit_expr(expr)),
        StmtKind::Empty => {}
        StmtKind::MacCall(mac) => {
            let MacCallStmt { mac, attrs, style: _, tokens: _ } = &**mac;
            walk_list!(visitor, visit_attribute, attrs);
            try_visit!(visitor.visit_mac_call(mac));
        }
    }
    V::Result::output()
}
