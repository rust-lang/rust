//! HIR walker for walking the contents of nodes.
//!
//! Here are the three available patterns for the visitor strategy,
//! in roughly the order of desirability:
//!
//! 1. **Shallow visit**: Get a simple callback for every item (or item-like thing) in the HIR.
//!    - Example: find all items with a `#[foo]` attribute on them.
//!    - How: Use the `hir_crate_items` or `hir_module_items` query to traverse over item-like ids
//!       (ItemId, TraitItemId, etc.) and use tcx.def_kind and `tcx.hir().item*(id)` to filter and
//!       access actual item-like thing, respectively.
//!    - Pro: Efficient; just walks the lists of item ids and gives users control whether to access
//!       the hir_owners themselves or not.
//!    - Con: Don't get information about nesting
//!    - Con: Don't have methods for specific bits of HIR, like "on
//!      every expr, do this".
//! 2. **Deep visit**: Want to scan for specific kinds of HIR nodes within
//!    an item, but don't care about how item-like things are nested
//!    within one another.
//!    - Example: Examine each expression to look for its type and do some check or other.
//!    - How: Implement `intravisit::Visitor` and override the `NestedFilter` type to
//!      `nested_filter::OnlyBodies` (and implement `nested_visit_map`), and use
//!      `tcx.hir().visit_all_item_likes_in_crate(&mut visitor)`. Within your
//!      `intravisit::Visitor` impl, implement methods like `visit_expr()` (don't forget to invoke
//!      `intravisit::walk_expr()` to keep walking the subparts).
//!    - Pro: Visitor methods for any kind of HIR node, not just item-like things.
//!    - Pro: Integrates well into dependency tracking.
//!    - Con: Don't get information about nesting between items
//! 3. **Nested visit**: Want to visit the whole HIR and you care about the nesting between
//!    item-like things.
//!    - Example: Lifetime resolution, which wants to bring lifetimes declared on the
//!      impl into scope while visiting the impl-items, and then back out again.
//!    - How: Implement `intravisit::Visitor` and override the `NestedFilter` type to
//!      `nested_filter::All` (and implement `nested_visit_map`). Walk your crate with
//!      `tcx.hir().walk_toplevel_module(visitor)` invoked on `tcx.hir().krate()`.
//!    - Pro: Visitor methods for any kind of HIR node, not just item-like things.
//!    - Pro: Preserves nesting information
//!    - Con: Does not integrate well into dependency tracking.
//!
//! If you have decided to use this visitor, here are some general
//! notes on how to do so:
//!
//! Each overridden visit method has full control over what
//! happens with its node, it can do its own traversal of the node's children,
//! call `intravisit::walk_*` to apply the default traversal algorithm, or prevent
//! deeper traversal by doing nothing.
//!
//! When visiting the HIR, the contents of nested items are NOT visited
//! by default. This is different from the AST visitor, which does a deep walk.
//! Hence this module is called `intravisit`; see the method `visit_nested_item`
//! for more details.
//!
//! Note: it is an important invariant that the default visitor walks
//! the body of a function in "execution order" - more concretely, if
//! we consider the reverse post-order (RPO) of the CFG implied by the HIR,
//! then a pre-order traversal of the HIR is consistent with the CFG RPO
//! on the *initial CFG point* of each HIR node, while a post-order traversal
//! of the HIR is consistent with the CFG RPO on each *final CFG point* of
//! each CFG node.
//!
//! One thing that follows is that if HIR node A always starts/ends executing
//! before HIR node B, then A appears in traversal pre/postorder before B,
//! respectively. (This follows from RPO respecting CFG domination).
//!
//! This order consistency is required in a few places in rustc, for
//! example coroutine inference, and possibly also HIR borrowck.

use rustc_ast::Label;
use rustc_ast::visit::{VisitorResult, try_visit, visit_opt, walk_list};
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::{Ident, Symbol};

use crate::hir::*;

pub trait IntoVisitor<'hir> {
    type Visitor: Visitor<'hir>;
    fn into_visitor(&self) -> Self::Visitor;
}

#[derive(Copy, Clone, Debug)]
pub enum FnKind<'a> {
    /// `#[xxx] pub async/const/extern "Abi" fn foo()`
    ItemFn(Ident, &'a Generics<'a>, FnHeader),

    /// `fn foo(&self)`
    Method(Ident, &'a FnSig<'a>),

    /// `|x, y| {}`
    Closure,
}

impl<'a> FnKind<'a> {
    pub fn header(&self) -> Option<&FnHeader> {
        match *self {
            FnKind::ItemFn(_, _, ref header) => Some(header),
            FnKind::Method(_, ref sig) => Some(&sig.header),
            FnKind::Closure => None,
        }
    }

    pub fn constness(self) -> Constness {
        self.header().map_or(Constness::NotConst, |header| header.constness)
    }

    pub fn asyncness(self) -> IsAsync {
        self.header().map_or(IsAsync::NotAsync, |header| header.asyncness)
    }
}

/// An abstract representation of the HIR `rustc_middle::hir::map::Map`.
pub trait Map<'hir> {
    /// Retrieves the `Node` corresponding to `id`.
    fn hir_node(&self, hir_id: HirId) -> Node<'hir>;
    fn hir_node_by_def_id(&self, def_id: LocalDefId) -> Node<'hir>;
    fn body(&self, id: BodyId) -> &'hir Body<'hir>;
    fn item(&self, id: ItemId) -> &'hir Item<'hir>;
    fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir>;
    fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir>;
    fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir>;
}

// Used when no map is actually available, forcing manual implementation of nested visitors.
impl<'hir> Map<'hir> for ! {
    fn hir_node(&self, _: HirId) -> Node<'hir> {
        *self;
    }
    fn hir_node_by_def_id(&self, _: LocalDefId) -> Node<'hir> {
        *self;
    }
    fn body(&self, _: BodyId) -> &'hir Body<'hir> {
        *self;
    }
    fn item(&self, _: ItemId) -> &'hir Item<'hir> {
        *self;
    }
    fn trait_item(&self, _: TraitItemId) -> &'hir TraitItem<'hir> {
        *self;
    }
    fn impl_item(&self, _: ImplItemId) -> &'hir ImplItem<'hir> {
        *self;
    }
    fn foreign_item(&self, _: ForeignItemId) -> &'hir ForeignItem<'hir> {
        *self;
    }
}

pub mod nested_filter {
    use super::Map;

    /// Specifies what nested things a visitor wants to visit. By "nested
    /// things", we are referring to bits of HIR that are not directly embedded
    /// within one another but rather indirectly, through a table in the crate.
    /// This is done to control dependencies during incremental compilation: the
    /// non-inline bits of HIR can be tracked and hashed separately.
    ///
    /// The most common choice is `OnlyBodies`, which will cause the visitor to
    /// visit fn bodies for fns that it encounters, and closure bodies, but
    /// skip over nested item-like things.
    ///
    /// See the comments at [`rustc_hir::intravisit`] for more details on the overall
    /// visit strategy.
    pub trait NestedFilter<'hir> {
        type Map: Map<'hir>;

        /// Whether the visitor visits nested "item-like" things.
        /// E.g., item, impl-item.
        const INTER: bool;
        /// Whether the visitor visits "intra item-like" things.
        /// E.g., function body, closure, `AnonConst`
        const INTRA: bool;
    }

    /// Do not visit any nested things. When you add a new
    /// "non-nested" thing, you will want to audit such uses to see if
    /// they remain valid.
    ///
    /// Use this if you are only walking some particular kind of tree
    /// (i.e., a type, or fn signature) and you don't want to thread a
    /// HIR map around.
    pub struct None(());
    impl NestedFilter<'_> for None {
        type Map = !;
        const INTER: bool = false;
        const INTRA: bool = false;
    }
}

use nested_filter::NestedFilter;

/// Each method of the Visitor trait is a hook to be potentially
/// overridden. Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_mod` method by default calls `intravisit::walk_mod`.
///
/// Note that this visitor does NOT visit nested items by default
/// (this is why the module is called `intravisit`, to distinguish it
/// from the AST's `visit` module, which acts differently). If you
/// simply want to visit all items in the crate in some order, you
/// should call `tcx.hir().visit_all_item_likes_in_crate`. Otherwise, see the comment
/// on `visit_nested_item` for details on how to visit nested items.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method. (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<'v>: Sized {
    // this type should not be overridden, it exists for convenient usage as `Self::Map`
    type Map: Map<'v> = <Self::NestedFilter as NestedFilter<'v>>::Map;

    ///////////////////////////////////////////////////////////////////////////
    // Nested items.

    /// Override this type to control which nested HIR are visited; see
    /// [`NestedFilter`] for details. If you override this type, you
    /// must also override [`nested_visit_map`](Self::nested_visit_map).
    ///
    /// **If for some reason you want the nested behavior, but don't
    /// have a `Map` at your disposal:** then override the
    /// `visit_nested_XXX` methods. If a new `visit_nested_XXX` variant is
    /// added in the future, it will cause a panic which can be detected
    /// and fixed appropriately.
    type NestedFilter: NestedFilter<'v> = nested_filter::None;

    /// The result type of the `visit_*` methods. Can be either `()`,
    /// or `ControlFlow<T>`.
    type Result: VisitorResult = ();

    /// If `type NestedFilter` is set to visit nested items, this method
    /// must also be overridden to provide a map to retrieve nested items.
    fn nested_visit_map(&mut self) -> Self::Map {
        panic!(
            "nested_visit_map must be implemented or consider using \
            `type NestedFilter = nested_filter::None` (the default)"
        );
    }

    /// Invoked when a nested item is encountered. By default, when
    /// `Self::NestedFilter` is `nested_filter::None`, this method does
    /// nothing. **You probably don't want to override this method** --
    /// instead, override [`Self::NestedFilter`] or use the "shallow" or
    /// "deep" visit patterns described at
    /// [`rustc_hir::intravisit`]. The only reason to override
    /// this method is if you want a nested pattern but cannot supply a
    /// [`Map`]; see `nested_visit_map` for advice.
    fn visit_nested_item(&mut self, id: ItemId) -> Self::Result {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().item(id);
            try_visit!(self.visit_item(item));
        }
        Self::Result::output()
    }

    /// Like `visit_nested_item()`, but for trait items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    fn visit_nested_trait_item(&mut self, id: TraitItemId) -> Self::Result {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().trait_item(id);
            try_visit!(self.visit_trait_item(item));
        }
        Self::Result::output()
    }

    /// Like `visit_nested_item()`, but for impl items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    fn visit_nested_impl_item(&mut self, id: ImplItemId) -> Self::Result {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().impl_item(id);
            try_visit!(self.visit_impl_item(item));
        }
        Self::Result::output()
    }

    /// Like `visit_nested_item()`, but for foreign items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    fn visit_nested_foreign_item(&mut self, id: ForeignItemId) -> Self::Result {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().foreign_item(id);
            try_visit!(self.visit_foreign_item(item));
        }
        Self::Result::output()
    }

    /// Invoked to visit the body of a function, method or closure. Like
    /// `visit_nested_item`, does nothing by default unless you override
    /// `Self::NestedFilter`.
    fn visit_nested_body(&mut self, id: BodyId) -> Self::Result {
        if Self::NestedFilter::INTRA {
            let body = self.nested_visit_map().body(id);
            try_visit!(self.visit_body(body));
        }
        Self::Result::output()
    }

    fn visit_param(&mut self, param: &'v Param<'v>) -> Self::Result {
        walk_param(self, param)
    }

    /// Visits the top-level item and (optionally) nested items / impl items. See
    /// `visit_nested_item` for details.
    fn visit_item(&mut self, i: &'v Item<'v>) -> Self::Result {
        walk_item(self, i)
    }

    fn visit_body(&mut self, b: &Body<'v>) -> Self::Result {
        walk_body(self, b)
    }

    ///////////////////////////////////////////////////////////////////////////

    fn visit_id(&mut self, _hir_id: HirId) -> Self::Result {
        Self::Result::output()
    }
    fn visit_name(&mut self, _name: Symbol) -> Self::Result {
        Self::Result::output()
    }
    fn visit_ident(&mut self, ident: Ident) -> Self::Result {
        walk_ident(self, ident)
    }
    fn visit_mod(&mut self, m: &'v Mod<'v>, _s: Span, n: HirId) -> Self::Result {
        walk_mod(self, m, n)
    }
    fn visit_foreign_item(&mut self, i: &'v ForeignItem<'v>) -> Self::Result {
        walk_foreign_item(self, i)
    }
    fn visit_local(&mut self, l: &'v LetStmt<'v>) -> Self::Result {
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'v Block<'v>) -> Self::Result {
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'v Stmt<'v>) -> Self::Result {
        walk_stmt(self, s)
    }
    fn visit_arm(&mut self, a: &'v Arm<'v>) -> Self::Result {
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'v Pat<'v>) -> Self::Result {
        walk_pat(self, p)
    }
    fn visit_pat_field(&mut self, f: &'v PatField<'v>) -> Self::Result {
        walk_pat_field(self, f)
    }
    fn visit_anon_const(&mut self, c: &'v AnonConst) -> Self::Result {
        walk_anon_const(self, c)
    }
    fn visit_inline_const(&mut self, c: &'v ConstBlock) -> Self::Result {
        walk_inline_const(self, c)
    }
    fn visit_const_arg(&mut self, c: &'v ConstArg<'v>) -> Self::Result {
        walk_const_arg(self, c)
    }
    fn visit_expr(&mut self, ex: &'v Expr<'v>) -> Self::Result {
        walk_expr(self, ex)
    }
    fn visit_expr_field(&mut self, field: &'v ExprField<'v>) -> Self::Result {
        walk_expr_field(self, field)
    }
    fn visit_ty(&mut self, t: &'v Ty<'v>) -> Self::Result {
        walk_ty(self, t)
    }
    fn visit_pattern_type_pattern(&mut self, _p: &'v Pat<'v>) {
        // Do nothing. Only a few visitors need to know the details of the pattern type,
        // and they opt into it. All other visitors will just choke on our fake patterns
        // because they aren't in a body.
    }
    fn visit_generic_param(&mut self, p: &'v GenericParam<'v>) -> Self::Result {
        walk_generic_param(self, p)
    }
    fn visit_const_param_default(&mut self, _param: HirId, ct: &'v ConstArg<'v>) -> Self::Result {
        walk_const_param_default(self, ct)
    }
    fn visit_generics(&mut self, g: &'v Generics<'v>) -> Self::Result {
        walk_generics(self, g)
    }
    fn visit_where_predicate(&mut self, predicate: &'v WherePredicate<'v>) -> Self::Result {
        walk_where_predicate(self, predicate)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'v FnRetTy<'v>) -> Self::Result {
        walk_fn_ret_ty(self, ret_ty)
    }
    fn visit_fn_decl(&mut self, fd: &'v FnDecl<'v>) -> Self::Result {
        walk_fn_decl(self, fd)
    }
    fn visit_fn(
        &mut self,
        fk: FnKind<'v>,
        fd: &'v FnDecl<'v>,
        b: BodyId,
        _: Span,
        id: LocalDefId,
    ) -> Self::Result {
        walk_fn(self, fk, fd, b, id)
    }
    fn visit_use(&mut self, path: &'v UsePath<'v>, hir_id: HirId) -> Self::Result {
        walk_use(self, path, hir_id)
    }
    fn visit_trait_item(&mut self, ti: &'v TraitItem<'v>) -> Self::Result {
        walk_trait_item(self, ti)
    }
    fn visit_trait_item_ref(&mut self, ii: &'v TraitItemRef) -> Self::Result {
        walk_trait_item_ref(self, ii)
    }
    fn visit_impl_item(&mut self, ii: &'v ImplItem<'v>) -> Self::Result {
        walk_impl_item(self, ii)
    }
    fn visit_foreign_item_ref(&mut self, ii: &'v ForeignItemRef) -> Self::Result {
        walk_foreign_item_ref(self, ii)
    }
    fn visit_impl_item_ref(&mut self, ii: &'v ImplItemRef) -> Self::Result {
        walk_impl_item_ref(self, ii)
    }
    fn visit_trait_ref(&mut self, t: &'v TraitRef<'v>) -> Self::Result {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'v GenericBound<'v>) -> Self::Result {
        walk_param_bound(self, bounds)
    }
    fn visit_precise_capturing_arg(&mut self, arg: &'v PreciseCapturingArg<'v>) -> Self::Result {
        walk_precise_capturing_arg(self, arg)
    }
    fn visit_poly_trait_ref(&mut self, t: &'v PolyTraitRef<'v>) -> Self::Result {
        walk_poly_trait_ref(self, t)
    }
    fn visit_opaque_ty(&mut self, opaque: &'v OpaqueTy<'v>) -> Self::Result {
        walk_opaque_ty(self, opaque)
    }
    fn visit_variant_data(&mut self, s: &'v VariantData<'v>) -> Self::Result {
        walk_struct_def(self, s)
    }
    fn visit_field_def(&mut self, s: &'v FieldDef<'v>) -> Self::Result {
        walk_field_def(self, s)
    }
    fn visit_enum_def(&mut self, enum_definition: &'v EnumDef<'v>, item_id: HirId) -> Self::Result {
        walk_enum_def(self, enum_definition, item_id)
    }
    fn visit_variant(&mut self, v: &'v Variant<'v>) -> Self::Result {
        walk_variant(self, v)
    }
    fn visit_label(&mut self, label: &'v Label) -> Self::Result {
        walk_label(self, label)
    }
    fn visit_infer(&mut self, inf: &'v InferArg) -> Self::Result {
        walk_inf(self, inf)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'v GenericArg<'v>) -> Self::Result {
        walk_generic_arg(self, generic_arg)
    }
    fn visit_lifetime(&mut self, lifetime: &'v Lifetime) -> Self::Result {
        walk_lifetime(self, lifetime)
    }
    // The span is that of the surrounding type/pattern/expr/whatever.
    fn visit_qpath(&mut self, qpath: &'v QPath<'v>, id: HirId, _span: Span) -> Self::Result {
        walk_qpath(self, qpath, id)
    }
    fn visit_path(&mut self, path: &Path<'v>, _id: HirId) -> Self::Result {
        walk_path(self, path)
    }
    fn visit_path_segment(&mut self, path_segment: &'v PathSegment<'v>) -> Self::Result {
        walk_path_segment(self, path_segment)
    }
    fn visit_generic_args(&mut self, generic_args: &'v GenericArgs<'v>) -> Self::Result {
        walk_generic_args(self, generic_args)
    }
    fn visit_assoc_item_constraint(
        &mut self,
        constraint: &'v AssocItemConstraint<'v>,
    ) -> Self::Result {
        walk_assoc_item_constraint(self, constraint)
    }
    fn visit_attribute(&mut self, _attr: &'v Attribute) -> Self::Result {
        Self::Result::output()
    }
    fn visit_associated_item_kind(&mut self, kind: &'v AssocItemKind) -> Self::Result {
        walk_associated_item_kind(self, kind)
    }
    fn visit_defaultness(&mut self, defaultness: &'v Defaultness) -> Self::Result {
        walk_defaultness(self, defaultness)
    }
    fn visit_inline_asm(&mut self, asm: &'v InlineAsm<'v>, id: HirId) -> Self::Result {
        walk_inline_asm(self, asm, id)
    }
}

pub fn walk_param<'v, V: Visitor<'v>>(visitor: &mut V, param: &'v Param<'v>) -> V::Result {
    try_visit!(visitor.visit_id(param.hir_id));
    visitor.visit_pat(param.pat)
}

pub fn walk_item<'v, V: Visitor<'v>>(visitor: &mut V, item: &'v Item<'v>) -> V::Result {
    try_visit!(visitor.visit_ident(item.ident));
    match item.kind {
        ItemKind::ExternCrate(orig_name) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            visit_opt!(visitor, visit_name, orig_name);
        }
        ItemKind::Use(ref path, _) => {
            try_visit!(visitor.visit_use(path, item.hir_id()));
        }
        ItemKind::Static(ref typ, _, body) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_ty(typ));
            try_visit!(visitor.visit_nested_body(body));
        }
        ItemKind::Const(ref typ, ref generics, body) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_ty(typ));
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_nested_body(body));
        }
        ItemKind::Fn(ref sig, ref generics, body_id) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_fn(
                FnKind::ItemFn(item.ident, generics, sig.header),
                sig.decl,
                body_id,
                item.span,
                item.owner_id.def_id,
            ));
        }
        ItemKind::Macro(..) => {
            try_visit!(visitor.visit_id(item.hir_id()));
        }
        ItemKind::Mod(ref module) => {
            // `visit_mod()` takes care of visiting the `Item`'s `HirId`.
            try_visit!(visitor.visit_mod(module, item.span, item.hir_id()));
        }
        ItemKind::ForeignMod { abi: _, items } => {
            try_visit!(visitor.visit_id(item.hir_id()));
            walk_list!(visitor, visit_foreign_item_ref, items);
        }
        ItemKind::GlobalAsm(asm) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_inline_asm(asm, item.hir_id()));
        }
        ItemKind::TyAlias(ref ty, ref generics) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_ty(ty));
            try_visit!(visitor.visit_generics(generics));
        }
        ItemKind::Enum(ref enum_definition, ref generics) => {
            try_visit!(visitor.visit_generics(generics));
            // `visit_enum_def()` takes care of visiting the `Item`'s `HirId`.
            try_visit!(visitor.visit_enum_def(enum_definition, item.hir_id()));
        }
        ItemKind::Impl(Impl {
            constness: _,
            safety: _,
            defaultness: _,
            polarity: _,
            defaultness_span: _,
            ref generics,
            ref of_trait,
            ref self_ty,
            items,
        }) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_generics(generics));
            visit_opt!(visitor, visit_trait_ref, of_trait);
            try_visit!(visitor.visit_ty(self_ty));
            walk_list!(visitor, visit_impl_item_ref, *items);
        }
        ItemKind::Struct(ref struct_definition, ref generics)
        | ItemKind::Union(ref struct_definition, ref generics) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_variant_data(struct_definition));
        }
        ItemKind::Trait(.., ref generics, bounds, trait_item_refs) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_trait_item_ref, trait_item_refs);
        }
        ItemKind::TraitAlias(ref generics, bounds) => {
            try_visit!(visitor.visit_id(item.hir_id()));
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds);
        }
    }
    V::Result::output()
}

pub fn walk_body<'v, V: Visitor<'v>>(visitor: &mut V, body: &Body<'v>) -> V::Result {
    walk_list!(visitor, visit_param, body.params);
    visitor.visit_expr(body.value)
}

pub fn walk_ident<'v, V: Visitor<'v>>(visitor: &mut V, ident: Ident) -> V::Result {
    visitor.visit_name(ident.name)
}

pub fn walk_mod<'v, V: Visitor<'v>>(
    visitor: &mut V,
    module: &'v Mod<'v>,
    mod_hir_id: HirId,
) -> V::Result {
    try_visit!(visitor.visit_id(mod_hir_id));
    walk_list!(visitor, visit_nested_item, module.item_ids.iter().copied());
    V::Result::output()
}

pub fn walk_foreign_item<'v, V: Visitor<'v>>(
    visitor: &mut V,
    foreign_item: &'v ForeignItem<'v>,
) -> V::Result {
    try_visit!(visitor.visit_id(foreign_item.hir_id()));
    try_visit!(visitor.visit_ident(foreign_item.ident));

    match foreign_item.kind {
        ForeignItemKind::Fn(ref sig, param_names, ref generics) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_fn_decl(sig.decl));
            walk_list!(visitor, visit_ident, param_names.iter().copied());
        }
        ForeignItemKind::Static(ref typ, _, _) => {
            try_visit!(visitor.visit_ty(typ));
        }
        ForeignItemKind::Type => (),
    }
    V::Result::output()
}

pub fn walk_local<'v, V: Visitor<'v>>(visitor: &mut V, local: &'v LetStmt<'v>) -> V::Result {
    // Intentionally visiting the expr first - the initialization expr
    // dominates the local's definition.
    visit_opt!(visitor, visit_expr, local.init);
    try_visit!(visitor.visit_id(local.hir_id));
    try_visit!(visitor.visit_pat(local.pat));
    visit_opt!(visitor, visit_block, local.els);
    visit_opt!(visitor, visit_ty, local.ty);
    V::Result::output()
}

pub fn walk_block<'v, V: Visitor<'v>>(visitor: &mut V, block: &'v Block<'v>) -> V::Result {
    try_visit!(visitor.visit_id(block.hir_id));
    walk_list!(visitor, visit_stmt, block.stmts);
    visit_opt!(visitor, visit_expr, block.expr);
    V::Result::output()
}

pub fn walk_stmt<'v, V: Visitor<'v>>(visitor: &mut V, statement: &'v Stmt<'v>) -> V::Result {
    try_visit!(visitor.visit_id(statement.hir_id));
    match statement.kind {
        StmtKind::Let(ref local) => visitor.visit_local(local),
        StmtKind::Item(item) => visitor.visit_nested_item(item),
        StmtKind::Expr(ref expression) | StmtKind::Semi(ref expression) => {
            visitor.visit_expr(expression)
        }
    }
}

pub fn walk_arm<'v, V: Visitor<'v>>(visitor: &mut V, arm: &'v Arm<'v>) -> V::Result {
    try_visit!(visitor.visit_id(arm.hir_id));
    try_visit!(visitor.visit_pat(arm.pat));
    visit_opt!(visitor, visit_expr, arm.guard);
    visitor.visit_expr(arm.body)
}

pub fn walk_pat<'v, V: Visitor<'v>>(visitor: &mut V, pattern: &'v Pat<'v>) -> V::Result {
    try_visit!(visitor.visit_id(pattern.hir_id));
    match pattern.kind {
        PatKind::TupleStruct(ref qpath, children, _) => {
            try_visit!(visitor.visit_qpath(qpath, pattern.hir_id, pattern.span));
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref qpath) => {
            try_visit!(visitor.visit_qpath(qpath, pattern.hir_id, pattern.span));
        }
        PatKind::Struct(ref qpath, fields, _) => {
            try_visit!(visitor.visit_qpath(qpath, pattern.hir_id, pattern.span));
            walk_list!(visitor, visit_pat_field, fields);
        }
        PatKind::Or(pats) => walk_list!(visitor, visit_pat, pats),
        PatKind::Tuple(tuple_elements, _) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatKind::Box(ref subpattern)
        | PatKind::Deref(ref subpattern)
        | PatKind::Ref(ref subpattern, _) => {
            try_visit!(visitor.visit_pat(subpattern));
        }
        PatKind::Binding(_, _hir_id, ident, ref optional_subpattern) => {
            try_visit!(visitor.visit_ident(ident));
            visit_opt!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => try_visit!(visitor.visit_expr(expression)),
        PatKind::Range(ref lower_bound, ref upper_bound, _) => {
            visit_opt!(visitor, visit_expr, lower_bound);
            visit_opt!(visitor, visit_expr, upper_bound);
        }
        PatKind::Never | PatKind::Wild | PatKind::Err(_) => (),
        PatKind::Slice(prepatterns, ref slice_pattern, postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            visit_opt!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
    }
    V::Result::output()
}

pub fn walk_pat_field<'v, V: Visitor<'v>>(visitor: &mut V, field: &'v PatField<'v>) -> V::Result {
    try_visit!(visitor.visit_id(field.hir_id));
    try_visit!(visitor.visit_ident(field.ident));
    visitor.visit_pat(field.pat)
}

pub fn walk_anon_const<'v, V: Visitor<'v>>(visitor: &mut V, constant: &'v AnonConst) -> V::Result {
    try_visit!(visitor.visit_id(constant.hir_id));
    visitor.visit_nested_body(constant.body)
}

pub fn walk_inline_const<'v, V: Visitor<'v>>(
    visitor: &mut V,
    constant: &'v ConstBlock,
) -> V::Result {
    try_visit!(visitor.visit_id(constant.hir_id));
    visitor.visit_nested_body(constant.body)
}

pub fn walk_const_arg<'v, V: Visitor<'v>>(
    visitor: &mut V,
    const_arg: &'v ConstArg<'v>,
) -> V::Result {
    try_visit!(visitor.visit_id(const_arg.hir_id));
    match &const_arg.kind {
        ConstArgKind::Path(qpath) => visitor.visit_qpath(qpath, const_arg.hir_id, qpath.span()),
        ConstArgKind::Anon(anon) => visitor.visit_anon_const(*anon),
        ConstArgKind::Infer(..) => V::Result::output(),
    }
}

pub fn walk_expr<'v, V: Visitor<'v>>(visitor: &mut V, expression: &'v Expr<'v>) -> V::Result {
    try_visit!(visitor.visit_id(expression.hir_id));
    match expression.kind {
        ExprKind::Array(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::ConstBlock(ref const_block) => {
            try_visit!(visitor.visit_inline_const(const_block))
        }
        ExprKind::Repeat(ref element, ref count) => {
            try_visit!(visitor.visit_expr(element));
            try_visit!(visitor.visit_const_arg(count));
        }
        ExprKind::Struct(ref qpath, fields, ref optional_base) => {
            try_visit!(visitor.visit_qpath(qpath, expression.hir_id, expression.span));
            walk_list!(visitor, visit_expr_field, fields);
            visit_opt!(visitor, visit_expr, optional_base);
        }
        ExprKind::Tup(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(ref callee_expression, arguments) => {
            try_visit!(visitor.visit_expr(callee_expression));
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(ref segment, receiver, arguments, _) => {
            try_visit!(visitor.visit_path_segment(segment));
            try_visit!(visitor.visit_expr(receiver));
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref left_expression, ref right_expression) => {
            try_visit!(visitor.visit_expr(left_expression));
            try_visit!(visitor.visit_expr(right_expression));
        }
        ExprKind::AddrOf(_, _, ref subexpression) | ExprKind::Unary(_, ref subexpression) => {
            try_visit!(visitor.visit_expr(subexpression));
        }
        ExprKind::Cast(ref subexpression, ref typ) | ExprKind::Type(ref subexpression, ref typ) => {
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_ty(typ));
        }
        ExprKind::DropTemps(ref subexpression) => {
            try_visit!(visitor.visit_expr(subexpression));
        }
        ExprKind::Let(LetExpr { span: _, pat, ty, init, recovered: _ }) => {
            // match the visit order in walk_local
            try_visit!(visitor.visit_expr(init));
            try_visit!(visitor.visit_pat(pat));
            visit_opt!(visitor, visit_ty, ty);
        }
        ExprKind::If(ref cond, ref then, ref else_opt) => {
            try_visit!(visitor.visit_expr(cond));
            try_visit!(visitor.visit_expr(then));
            visit_opt!(visitor, visit_expr, else_opt);
        }
        ExprKind::Loop(ref block, ref opt_label, _, _) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::Match(ref subexpression, arms, _) => {
            try_visit!(visitor.visit_expr(subexpression));
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(&Closure {
            def_id,
            binder: _,
            bound_generic_params,
            fn_decl,
            body,
            capture_clause: _,
            fn_decl_span: _,
            fn_arg_span: _,
            kind: _,
            constness: _,
        }) => {
            walk_list!(visitor, visit_generic_param, bound_generic_params);
            try_visit!(visitor.visit_fn(FnKind::Closure, fn_decl, body, expression.span, def_id));
        }
        ExprKind::Block(ref block, ref opt_label) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::Assign(ref lhs, ref rhs, _) => {
            try_visit!(visitor.visit_expr(rhs));
            try_visit!(visitor.visit_expr(lhs));
        }
        ExprKind::AssignOp(_, ref left_expression, ref right_expression) => {
            try_visit!(visitor.visit_expr(right_expression));
            try_visit!(visitor.visit_expr(left_expression));
        }
        ExprKind::Field(ref subexpression, ident) => {
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_ident(ident));
        }
        ExprKind::Index(ref main_expression, ref index_expression, _) => {
            try_visit!(visitor.visit_expr(main_expression));
            try_visit!(visitor.visit_expr(index_expression));
        }
        ExprKind::Path(ref qpath) => {
            try_visit!(visitor.visit_qpath(qpath, expression.hir_id, expression.span));
        }
        ExprKind::Break(ref destination, ref opt_expr) => {
            visit_opt!(visitor, visit_label, &destination.label);
            visit_opt!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(ref destination) => {
            visit_opt!(visitor, visit_label, &destination.label);
        }
        ExprKind::Ret(ref optional_expression) => {
            visit_opt!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Become(ref expr) => try_visit!(visitor.visit_expr(expr)),
        ExprKind::InlineAsm(ref asm) => {
            try_visit!(visitor.visit_inline_asm(asm, expression.hir_id));
        }
        ExprKind::OffsetOf(ref container, ref fields) => {
            try_visit!(visitor.visit_ty(container));
            walk_list!(visitor, visit_ident, fields.iter().copied());
        }
        ExprKind::Yield(ref subexpression, _) => {
            try_visit!(visitor.visit_expr(subexpression));
        }
        ExprKind::Lit(_) | ExprKind::Err(_) => {}
    }
    V::Result::output()
}

pub fn walk_expr_field<'v, V: Visitor<'v>>(visitor: &mut V, field: &'v ExprField<'v>) -> V::Result {
    try_visit!(visitor.visit_id(field.hir_id));
    try_visit!(visitor.visit_ident(field.ident));
    visitor.visit_expr(field.expr)
}

pub fn walk_ty<'v, V: Visitor<'v>>(visitor: &mut V, typ: &'v Ty<'v>) -> V::Result {
    try_visit!(visitor.visit_id(typ.hir_id));

    match typ.kind {
        TyKind::Slice(ref ty) => try_visit!(visitor.visit_ty(ty)),
        TyKind::Ptr(ref mutable_type) => try_visit!(visitor.visit_ty(mutable_type.ty)),
        TyKind::Ref(ref lifetime, ref mutable_type) => {
            try_visit!(visitor.visit_lifetime(lifetime));
            try_visit!(visitor.visit_ty(mutable_type.ty));
        }
        TyKind::Never => {}
        TyKind::Tup(tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(ref function_declaration) => {
            walk_list!(visitor, visit_generic_param, function_declaration.generic_params);
            try_visit!(visitor.visit_fn_decl(function_declaration.decl));
        }
        TyKind::Path(ref qpath) => {
            try_visit!(visitor.visit_qpath(qpath, typ.hir_id, typ.span));
        }
        TyKind::OpaqueDef(opaque) => {
            try_visit!(visitor.visit_opaque_ty(opaque));
        }
        TyKind::Array(ref ty, ref length) => {
            try_visit!(visitor.visit_ty(ty));
            try_visit!(visitor.visit_const_arg(length));
        }
        TyKind::TraitObject(bounds, ref lifetime, _syntax) => {
            for bound in bounds {
                try_visit!(visitor.visit_poly_trait_ref(bound));
            }
            try_visit!(visitor.visit_lifetime(lifetime));
        }
        TyKind::Typeof(ref expression) => try_visit!(visitor.visit_anon_const(expression)),
        TyKind::Infer | TyKind::InferDelegation(..) | TyKind::Err(_) => {}
        TyKind::AnonAdt(item_id) => {
            try_visit!(visitor.visit_nested_item(item_id));
        }
        TyKind::Pat(ty, pat) => {
            try_visit!(visitor.visit_ty(ty));
            try_visit!(visitor.visit_pattern_type_pattern(pat));
        }
    }
    V::Result::output()
}

pub fn walk_generic_param<'v, V: Visitor<'v>>(
    visitor: &mut V,
    param: &'v GenericParam<'v>,
) -> V::Result {
    try_visit!(visitor.visit_id(param.hir_id));
    match param.name {
        ParamName::Plain(ident) => try_visit!(visitor.visit_ident(ident)),
        ParamName::Error | ParamName::Fresh => {}
    }
    match param.kind {
        GenericParamKind::Lifetime { .. } => {}
        GenericParamKind::Type { ref default, .. } => visit_opt!(visitor, visit_ty, default),
        GenericParamKind::Const { ref ty, ref default, synthetic: _ } => {
            try_visit!(visitor.visit_ty(ty));
            if let Some(ref default) = default {
                try_visit!(visitor.visit_const_param_default(param.hir_id, default));
            }
        }
    }
    V::Result::output()
}

pub fn walk_const_param_default<'v, V: Visitor<'v>>(
    visitor: &mut V,
    ct: &'v ConstArg<'v>,
) -> V::Result {
    visitor.visit_const_arg(ct)
}

pub fn walk_generics<'v, V: Visitor<'v>>(visitor: &mut V, generics: &'v Generics<'v>) -> V::Result {
    walk_list!(visitor, visit_generic_param, generics.params);
    walk_list!(visitor, visit_where_predicate, generics.predicates);
    V::Result::output()
}

pub fn walk_where_predicate<'v, V: Visitor<'v>>(
    visitor: &mut V,
    predicate: &'v WherePredicate<'v>,
) -> V::Result {
    let &WherePredicate { hir_id, kind, span: _ } = predicate;
    try_visit!(visitor.visit_id(hir_id));
    match *kind {
        WherePredicateKind::BoundPredicate(WhereBoundPredicate {
            ref bounded_ty,
            bounds,
            bound_generic_params,
            origin: _,
        }) => {
            try_visit!(visitor.visit_ty(bounded_ty));
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicateKind::RegionPredicate(WhereRegionPredicate {
            ref lifetime,
            bounds,
            in_where_clause: _,
        }) => {
            try_visit!(visitor.visit_lifetime(lifetime));
            walk_list!(visitor, visit_param_bound, bounds);
        }
        WherePredicateKind::EqPredicate(WhereEqPredicate { ref lhs_ty, ref rhs_ty }) => {
            try_visit!(visitor.visit_ty(lhs_ty));
            try_visit!(visitor.visit_ty(rhs_ty));
        }
    }
    V::Result::output()
}

pub fn walk_fn_decl<'v, V: Visitor<'v>>(
    visitor: &mut V,
    function_declaration: &'v FnDecl<'v>,
) -> V::Result {
    walk_list!(visitor, visit_ty, function_declaration.inputs);
    visitor.visit_fn_ret_ty(&function_declaration.output)
}

pub fn walk_fn_ret_ty<'v, V: Visitor<'v>>(visitor: &mut V, ret_ty: &'v FnRetTy<'v>) -> V::Result {
    if let FnRetTy::Return(output_ty) = *ret_ty {
        try_visit!(visitor.visit_ty(output_ty));
    }
    V::Result::output()
}

pub fn walk_fn<'v, V: Visitor<'v>>(
    visitor: &mut V,
    function_kind: FnKind<'v>,
    function_declaration: &'v FnDecl<'v>,
    body_id: BodyId,
    _: LocalDefId,
) -> V::Result {
    try_visit!(visitor.visit_fn_decl(function_declaration));
    try_visit!(walk_fn_kind(visitor, function_kind));
    visitor.visit_nested_body(body_id)
}

pub fn walk_fn_kind<'v, V: Visitor<'v>>(visitor: &mut V, function_kind: FnKind<'v>) -> V::Result {
    match function_kind {
        FnKind::ItemFn(_, generics, ..) => {
            try_visit!(visitor.visit_generics(generics));
        }
        FnKind::Closure | FnKind::Method(..) => {}
    }
    V::Result::output()
}

pub fn walk_use<'v, V: Visitor<'v>>(
    visitor: &mut V,
    path: &'v UsePath<'v>,
    hir_id: HirId,
) -> V::Result {
    try_visit!(visitor.visit_id(hir_id));
    let UsePath { segments, ref res, span } = *path;
    for &res in res {
        try_visit!(visitor.visit_path(&Path { segments, res, span }, hir_id));
    }
    V::Result::output()
}

pub fn walk_trait_item<'v, V: Visitor<'v>>(
    visitor: &mut V,
    trait_item: &'v TraitItem<'v>,
) -> V::Result {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let TraitItem { ident, generics, ref defaultness, ref kind, span, owner_id: _ } = *trait_item;
    let hir_id = trait_item.hir_id();
    try_visit!(visitor.visit_ident(ident));
    try_visit!(visitor.visit_generics(&generics));
    try_visit!(visitor.visit_defaultness(&defaultness));
    try_visit!(visitor.visit_id(hir_id));
    match *kind {
        TraitItemKind::Const(ref ty, default) => {
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_nested_body, default);
        }
        TraitItemKind::Fn(ref sig, TraitFn::Required(param_names)) => {
            try_visit!(visitor.visit_fn_decl(sig.decl));
            walk_list!(visitor, visit_ident, param_names.iter().copied());
        }
        TraitItemKind::Fn(ref sig, TraitFn::Provided(body_id)) => {
            try_visit!(visitor.visit_fn(
                FnKind::Method(ident, sig),
                sig.decl,
                body_id,
                span,
                trait_item.owner_id.def_id,
            ));
        }
        TraitItemKind::Type(bounds, ref default) => {
            walk_list!(visitor, visit_param_bound, bounds);
            visit_opt!(visitor, visit_ty, default);
        }
    }
    V::Result::output()
}

pub fn walk_trait_item_ref<'v, V: Visitor<'v>>(
    visitor: &mut V,
    trait_item_ref: &'v TraitItemRef,
) -> V::Result {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let TraitItemRef { id, ident, ref kind, span: _ } = *trait_item_ref;
    try_visit!(visitor.visit_nested_trait_item(id));
    try_visit!(visitor.visit_ident(ident));
    visitor.visit_associated_item_kind(kind)
}

pub fn walk_impl_item<'v, V: Visitor<'v>>(
    visitor: &mut V,
    impl_item: &'v ImplItem<'v>,
) -> V::Result {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ImplItem {
        owner_id: _,
        ident,
        ref generics,
        ref kind,
        ref defaultness,
        span: _,
        vis_span: _,
    } = *impl_item;

    try_visit!(visitor.visit_ident(ident));
    try_visit!(visitor.visit_generics(generics));
    try_visit!(visitor.visit_defaultness(defaultness));
    try_visit!(visitor.visit_id(impl_item.hir_id()));
    match *kind {
        ImplItemKind::Const(ref ty, body) => {
            try_visit!(visitor.visit_ty(ty));
            visitor.visit_nested_body(body)
        }
        ImplItemKind::Fn(ref sig, body_id) => visitor.visit_fn(
            FnKind::Method(impl_item.ident, sig),
            sig.decl,
            body_id,
            impl_item.span,
            impl_item.owner_id.def_id,
        ),
        ImplItemKind::Type(ref ty) => visitor.visit_ty(ty),
    }
}

pub fn walk_foreign_item_ref<'v, V: Visitor<'v>>(
    visitor: &mut V,
    foreign_item_ref: &'v ForeignItemRef,
) -> V::Result {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ForeignItemRef { id, ident, span: _ } = *foreign_item_ref;
    try_visit!(visitor.visit_nested_foreign_item(id));
    visitor.visit_ident(ident)
}

pub fn walk_impl_item_ref<'v, V: Visitor<'v>>(
    visitor: &mut V,
    impl_item_ref: &'v ImplItemRef,
) -> V::Result {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ImplItemRef { id, ident, ref kind, span: _, trait_item_def_id: _ } = *impl_item_ref;
    try_visit!(visitor.visit_nested_impl_item(id));
    try_visit!(visitor.visit_ident(ident));
    visitor.visit_associated_item_kind(kind)
}

pub fn walk_trait_ref<'v, V: Visitor<'v>>(
    visitor: &mut V,
    trait_ref: &'v TraitRef<'v>,
) -> V::Result {
    try_visit!(visitor.visit_id(trait_ref.hir_ref_id));
    visitor.visit_path(trait_ref.path, trait_ref.hir_ref_id)
}

pub fn walk_param_bound<'v, V: Visitor<'v>>(
    visitor: &mut V,
    bound: &'v GenericBound<'v>,
) -> V::Result {
    match *bound {
        GenericBound::Trait(ref typ) => visitor.visit_poly_trait_ref(typ),
        GenericBound::Outlives(ref lifetime) => visitor.visit_lifetime(lifetime),
        GenericBound::Use(args, _) => {
            walk_list!(visitor, visit_precise_capturing_arg, args);
            V::Result::output()
        }
    }
}

pub fn walk_precise_capturing_arg<'v, V: Visitor<'v>>(
    visitor: &mut V,
    arg: &'v PreciseCapturingArg<'v>,
) -> V::Result {
    match *arg {
        PreciseCapturingArg::Lifetime(lt) => visitor.visit_lifetime(lt),
        PreciseCapturingArg::Param(param) => visitor.visit_id(param.hir_id),
    }
}

pub fn walk_poly_trait_ref<'v, V: Visitor<'v>>(
    visitor: &mut V,
    trait_ref: &'v PolyTraitRef<'v>,
) -> V::Result {
    walk_list!(visitor, visit_generic_param, trait_ref.bound_generic_params);
    visitor.visit_trait_ref(&trait_ref.trait_ref)
}

pub fn walk_opaque_ty<'v, V: Visitor<'v>>(visitor: &mut V, opaque: &'v OpaqueTy<'v>) -> V::Result {
    let &OpaqueTy { hir_id, def_id: _, bounds, origin: _, span: _ } = opaque;
    try_visit!(visitor.visit_id(hir_id));
    walk_list!(visitor, visit_param_bound, bounds);
    V::Result::output()
}

pub fn walk_struct_def<'v, V: Visitor<'v>>(
    visitor: &mut V,
    struct_definition: &'v VariantData<'v>,
) -> V::Result {
    visit_opt!(visitor, visit_id, struct_definition.ctor_hir_id());
    walk_list!(visitor, visit_field_def, struct_definition.fields());
    V::Result::output()
}

pub fn walk_field_def<'v, V: Visitor<'v>>(visitor: &mut V, field: &'v FieldDef<'v>) -> V::Result {
    try_visit!(visitor.visit_id(field.hir_id));
    try_visit!(visitor.visit_ident(field.ident));
    visitor.visit_ty(field.ty)
}

pub fn walk_enum_def<'v, V: Visitor<'v>>(
    visitor: &mut V,
    enum_definition: &'v EnumDef<'v>,
    item_id: HirId,
) -> V::Result {
    try_visit!(visitor.visit_id(item_id));
    walk_list!(visitor, visit_variant, enum_definition.variants);
    V::Result::output()
}

pub fn walk_variant<'v, V: Visitor<'v>>(visitor: &mut V, variant: &'v Variant<'v>) -> V::Result {
    try_visit!(visitor.visit_ident(variant.ident));
    try_visit!(visitor.visit_id(variant.hir_id));
    try_visit!(visitor.visit_variant_data(&variant.data));
    visit_opt!(visitor, visit_anon_const, &variant.disr_expr);
    V::Result::output()
}

pub fn walk_label<'v, V: Visitor<'v>>(visitor: &mut V, label: &'v Label) -> V::Result {
    visitor.visit_ident(label.ident)
}

pub fn walk_inf<'v, V: Visitor<'v>>(visitor: &mut V, inf: &'v InferArg) -> V::Result {
    visitor.visit_id(inf.hir_id)
}

pub fn walk_generic_arg<'v, V: Visitor<'v>>(
    visitor: &mut V,
    generic_arg: &'v GenericArg<'v>,
) -> V::Result {
    match generic_arg {
        GenericArg::Lifetime(lt) => visitor.visit_lifetime(lt),
        GenericArg::Type(ty) => visitor.visit_ty(ty),
        GenericArg::Const(ct) => visitor.visit_const_arg(ct),
        GenericArg::Infer(inf) => visitor.visit_infer(inf),
    }
}

pub fn walk_lifetime<'v, V: Visitor<'v>>(visitor: &mut V, lifetime: &'v Lifetime) -> V::Result {
    try_visit!(visitor.visit_id(lifetime.hir_id));
    visitor.visit_ident(lifetime.ident)
}

pub fn walk_qpath<'v, V: Visitor<'v>>(
    visitor: &mut V,
    qpath: &'v QPath<'v>,
    id: HirId,
) -> V::Result {
    match *qpath {
        QPath::Resolved(ref maybe_qself, ref path) => {
            visit_opt!(visitor, visit_ty, maybe_qself);
            visitor.visit_path(path, id)
        }
        QPath::TypeRelative(ref qself, ref segment) => {
            try_visit!(visitor.visit_ty(qself));
            visitor.visit_path_segment(segment)
        }
        QPath::LangItem(..) => V::Result::output(),
    }
}

pub fn walk_path<'v, V: Visitor<'v>>(visitor: &mut V, path: &Path<'v>) -> V::Result {
    walk_list!(visitor, visit_path_segment, path.segments);
    V::Result::output()
}

pub fn walk_path_segment<'v, V: Visitor<'v>>(
    visitor: &mut V,
    segment: &'v PathSegment<'v>,
) -> V::Result {
    try_visit!(visitor.visit_ident(segment.ident));
    try_visit!(visitor.visit_id(segment.hir_id));
    visit_opt!(visitor, visit_generic_args, segment.args);
    V::Result::output()
}

pub fn walk_generic_args<'v, V: Visitor<'v>>(
    visitor: &mut V,
    generic_args: &'v GenericArgs<'v>,
) -> V::Result {
    walk_list!(visitor, visit_generic_arg, generic_args.args);
    walk_list!(visitor, visit_assoc_item_constraint, generic_args.constraints);
    V::Result::output()
}

pub fn walk_assoc_item_constraint<'v, V: Visitor<'v>>(
    visitor: &mut V,
    constraint: &'v AssocItemConstraint<'v>,
) -> V::Result {
    try_visit!(visitor.visit_id(constraint.hir_id));
    try_visit!(visitor.visit_ident(constraint.ident));
    try_visit!(visitor.visit_generic_args(constraint.gen_args));
    match constraint.kind {
        AssocItemConstraintKind::Equality { ref term } => match term {
            Term::Ty(ref ty) => try_visit!(visitor.visit_ty(ty)),
            Term::Const(ref c) => try_visit!(visitor.visit_const_arg(c)),
        },
        AssocItemConstraintKind::Bound { bounds } => {
            walk_list!(visitor, visit_param_bound, bounds)
        }
    }
    V::Result::output()
}

pub fn walk_associated_item_kind<'v, V: Visitor<'v>>(_: &mut V, _: &'v AssocItemKind) -> V::Result {
    // No visitable content here: this fn exists so you can call it if
    // the right thing to do, should content be added in the future,
    // would be to walk it.
    V::Result::output()
}

pub fn walk_defaultness<'v, V: Visitor<'v>>(_: &mut V, _: &'v Defaultness) -> V::Result {
    // No visitable content here: this fn exists so you can call it if
    // the right thing to do, should content be added in the future,
    // would be to walk it.
    V::Result::output()
}

pub fn walk_inline_asm<'v, V: Visitor<'v>>(
    visitor: &mut V,
    asm: &'v InlineAsm<'v>,
    id: HirId,
) -> V::Result {
    for (op, op_sp) in asm.operands {
        match op {
            InlineAsmOperand::In { expr, .. } | InlineAsmOperand::InOut { expr, .. } => {
                try_visit!(visitor.visit_expr(expr));
            }
            InlineAsmOperand::Out { expr, .. } => {
                visit_opt!(visitor, visit_expr, expr);
            }
            InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                try_visit!(visitor.visit_expr(in_expr));
                visit_opt!(visitor, visit_expr, out_expr);
            }
            InlineAsmOperand::Const { anon_const, .. }
            | InlineAsmOperand::SymFn { anon_const, .. } => {
                try_visit!(visitor.visit_anon_const(anon_const));
            }
            InlineAsmOperand::SymStatic { path, .. } => {
                try_visit!(visitor.visit_qpath(path, id, *op_sp));
            }
            InlineAsmOperand::Label { block } => try_visit!(visitor.visit_block(block)),
        }
    }
    V::Result::output()
}
