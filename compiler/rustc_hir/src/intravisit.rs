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
//! example generator inference, and possibly also HIR borrowck.

use crate::hir::*;
use rustc_ast::walk_list;
use rustc_ast::{Attribute, Label};
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

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
    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    fn find(&self, hir_id: HirId) -> Option<Node<'hir>>;
    fn body(&self, id: BodyId) -> &'hir Body<'hir>;
    fn item(&self, id: ItemId) -> &'hir Item<'hir>;
    fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir>;
    fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir>;
    fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir>;
}

// Used when no map is actually available, forcing manual implementation of nested visitors.
impl<'hir> Map<'hir> for ! {
    fn find(&self, _: HirId) -> Option<Node<'hir>> {
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
    /// See the comments on `ItemLikeVisitor` for more details on the overall
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
    /// "deep" visit patterns described on
    /// `itemlikevisit::ItemLikeVisitor`. The only reason to override
    /// this method is if you want a nested pattern but cannot supply a
    /// [`Map`]; see `nested_visit_map` for advice.
    fn visit_nested_item(&mut self, id: ItemId) {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().item(id);
            self.visit_item(item);
        }
    }

    /// Like `visit_nested_item()`, but for trait items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    fn visit_nested_trait_item(&mut self, id: TraitItemId) {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().trait_item(id);
            self.visit_trait_item(item);
        }
    }

    /// Like `visit_nested_item()`, but for impl items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    fn visit_nested_impl_item(&mut self, id: ImplItemId) {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().impl_item(id);
            self.visit_impl_item(item);
        }
    }

    /// Like `visit_nested_item()`, but for foreign items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    fn visit_nested_foreign_item(&mut self, id: ForeignItemId) {
        if Self::NestedFilter::INTER {
            let item = self.nested_visit_map().foreign_item(id);
            self.visit_foreign_item(item);
        }
    }

    /// Invoked to visit the body of a function, method or closure. Like
    /// `visit_nested_item`, does nothing by default unless you override
    /// `Self::NestedFilter`.
    fn visit_nested_body(&mut self, id: BodyId) {
        if Self::NestedFilter::INTRA {
            let body = self.nested_visit_map().body(id);
            self.visit_body(body);
        }
    }

    fn visit_param(&mut self, param: &'v Param<'v>) {
        walk_param(self, param)
    }

    /// Visits the top-level item and (optionally) nested items / impl items. See
    /// `visit_nested_item` for details.
    fn visit_item(&mut self, i: &'v Item<'v>) {
        walk_item(self, i)
    }

    fn visit_body(&mut self, b: &'v Body<'v>) {
        walk_body(self, b);
    }

    ///////////////////////////////////////////////////////////////////////////

    fn visit_id(&mut self, _hir_id: HirId) {
        // Nothing to do.
    }
    fn visit_name(&mut self, _name: Symbol) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, ident: Ident) {
        walk_ident(self, ident)
    }
    fn visit_mod(&mut self, m: &'v Mod<'v>, _s: Span, n: HirId) {
        walk_mod(self, m, n)
    }
    fn visit_foreign_item(&mut self, i: &'v ForeignItem<'v>) {
        walk_foreign_item(self, i)
    }
    fn visit_local(&mut self, l: &'v Local<'v>) {
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'v Block<'v>) {
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'v Stmt<'v>) {
        walk_stmt(self, s)
    }
    fn visit_arm(&mut self, a: &'v Arm<'v>) {
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'v Pat<'v>) {
        walk_pat(self, p)
    }
    fn visit_pat_field(&mut self, f: &'v PatField<'v>) {
        walk_pat_field(self, f)
    }
    fn visit_array_length(&mut self, len: &'v ArrayLen) {
        walk_array_len(self, len)
    }
    fn visit_anon_const(&mut self, c: &'v AnonConst) {
        walk_anon_const(self, c)
    }
    fn visit_inline_const(&mut self, c: &'v ConstBlock) {
        walk_inline_const(self, c)
    }
    fn visit_expr(&mut self, ex: &'v Expr<'v>) {
        walk_expr(self, ex)
    }
    fn visit_let_expr(&mut self, lex: &'v Let<'v>) {
        walk_let_expr(self, lex)
    }
    fn visit_expr_field(&mut self, field: &'v ExprField<'v>) {
        walk_expr_field(self, field)
    }
    fn visit_ty(&mut self, t: &'v Ty<'v>) {
        walk_ty(self, t)
    }
    fn visit_generic_param(&mut self, p: &'v GenericParam<'v>) {
        walk_generic_param(self, p)
    }
    fn visit_const_param_default(&mut self, _param: HirId, ct: &'v AnonConst) {
        walk_const_param_default(self, ct)
    }
    fn visit_generics(&mut self, g: &'v Generics<'v>) {
        walk_generics(self, g)
    }
    fn visit_where_predicate(&mut self, predicate: &'v WherePredicate<'v>) {
        walk_where_predicate(self, predicate)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'v FnRetTy<'v>) {
        walk_fn_ret_ty(self, ret_ty)
    }
    fn visit_fn_decl(&mut self, fd: &'v FnDecl<'v>) {
        walk_fn_decl(self, fd)
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl<'v>, b: BodyId, _: Span, id: LocalDefId) {
        walk_fn(self, fk, fd, b, id)
    }
    fn visit_use(&mut self, path: &'v UsePath<'v>, hir_id: HirId) {
        walk_use(self, path, hir_id)
    }
    fn visit_trait_item(&mut self, ti: &'v TraitItem<'v>) {
        walk_trait_item(self, ti)
    }
    fn visit_trait_item_ref(&mut self, ii: &'v TraitItemRef) {
        walk_trait_item_ref(self, ii)
    }
    fn visit_impl_item(&mut self, ii: &'v ImplItem<'v>) {
        walk_impl_item(self, ii)
    }
    fn visit_foreign_item_ref(&mut self, ii: &'v ForeignItemRef) {
        walk_foreign_item_ref(self, ii)
    }
    fn visit_impl_item_ref(&mut self, ii: &'v ImplItemRef) {
        walk_impl_item_ref(self, ii)
    }
    fn visit_trait_ref(&mut self, t: &'v TraitRef<'v>) {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'v GenericBound<'v>) {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'v PolyTraitRef<'v>) {
        walk_poly_trait_ref(self, t)
    }
    fn visit_variant_data(&mut self, s: &'v VariantData<'v>) {
        walk_struct_def(self, s)
    }
    fn visit_field_def(&mut self, s: &'v FieldDef<'v>) {
        walk_field_def(self, s)
    }
    fn visit_enum_def(&mut self, enum_definition: &'v EnumDef<'v>, item_id: HirId) {
        walk_enum_def(self, enum_definition, item_id)
    }
    fn visit_variant(&mut self, v: &'v Variant<'v>) {
        walk_variant(self, v)
    }
    fn visit_label(&mut self, label: &'v Label) {
        walk_label(self, label)
    }
    fn visit_infer(&mut self, inf: &'v InferArg) {
        walk_inf(self, inf);
    }
    fn visit_generic_arg(&mut self, generic_arg: &'v GenericArg<'v>) {
        walk_generic_arg(self, generic_arg);
    }
    fn visit_lifetime(&mut self, lifetime: &'v Lifetime) {
        walk_lifetime(self, lifetime)
    }
    // The span is that of the surrounding type/pattern/expr/whatever.
    fn visit_qpath(&mut self, qpath: &'v QPath<'v>, id: HirId, _span: Span) {
        walk_qpath(self, qpath, id)
    }
    fn visit_path(&mut self, path: &Path<'v>, _id: HirId) {
        walk_path(self, path)
    }
    fn visit_path_segment(&mut self, path_segment: &'v PathSegment<'v>) {
        walk_path_segment(self, path_segment)
    }
    fn visit_generic_args(&mut self, generic_args: &'v GenericArgs<'v>) {
        walk_generic_args(self, generic_args)
    }
    fn visit_assoc_type_binding(&mut self, type_binding: &'v TypeBinding<'v>) {
        walk_assoc_type_binding(self, type_binding)
    }
    fn visit_attribute(&mut self, _attr: &'v Attribute) {}
    fn visit_associated_item_kind(&mut self, kind: &'v AssocItemKind) {
        walk_associated_item_kind(self, kind);
    }
    fn visit_defaultness(&mut self, defaultness: &'v Defaultness) {
        walk_defaultness(self, defaultness);
    }
    fn visit_inline_asm(&mut self, asm: &'v InlineAsm<'v>, id: HirId) {
        walk_inline_asm(self, asm, id);
    }
}

pub fn walk_param<'v, V: Visitor<'v>>(visitor: &mut V, param: &'v Param<'v>) {
    visitor.visit_id(param.hir_id);
    visitor.visit_pat(param.pat);
}

pub fn walk_item<'v, V: Visitor<'v>>(visitor: &mut V, item: &'v Item<'v>) {
    visitor.visit_ident(item.ident);
    match item.kind {
        ItemKind::ExternCrate(orig_name) => {
            visitor.visit_id(item.hir_id());
            if let Some(orig_name) = orig_name {
                visitor.visit_name(orig_name);
            }
        }
        ItemKind::Use(ref path, _) => {
            visitor.visit_use(path, item.hir_id());
        }
        ItemKind::Static(ref typ, _, body) | ItemKind::Const(ref typ, body) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_ty(typ);
            visitor.visit_nested_body(body);
        }
        ItemKind::Fn(ref sig, ref generics, body_id) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_fn(
                FnKind::ItemFn(item.ident, generics, sig.header),
                sig.decl,
                body_id,
                item.span,
                item.owner_id.def_id,
            )
        }
        ItemKind::Macro(..) => {
            visitor.visit_id(item.hir_id());
        }
        ItemKind::Mod(ref module) => {
            // `visit_mod()` takes care of visiting the `Item`'s `HirId`.
            visitor.visit_mod(module, item.span, item.hir_id())
        }
        ItemKind::ForeignMod { abi: _, items } => {
            visitor.visit_id(item.hir_id());
            walk_list!(visitor, visit_foreign_item_ref, items);
        }
        ItemKind::GlobalAsm(asm) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_inline_asm(asm, item.hir_id());
        }
        ItemKind::TyAlias(ref ty, ref generics) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_ty(ty);
            visitor.visit_generics(generics)
        }
        ItemKind::OpaqueTy(OpaqueTy { ref generics, bounds, .. }) => {
            visitor.visit_id(item.hir_id());
            walk_generics(visitor, generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ItemKind::Enum(ref enum_definition, ref generics) => {
            visitor.visit_generics(generics);
            // `visit_enum_def()` takes care of visiting the `Item`'s `HirId`.
            visitor.visit_enum_def(enum_definition, item.hir_id())
        }
        ItemKind::Impl(Impl {
            unsafety: _,
            defaultness: _,
            polarity: _,
            constness: _,
            defaultness_span: _,
            ref generics,
            ref of_trait,
            ref self_ty,
            items,
        }) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_trait_ref, of_trait);
            visitor.visit_ty(self_ty);
            walk_list!(visitor, visit_impl_item_ref, *items);
        }
        ItemKind::Struct(ref struct_definition, ref generics)
        | ItemKind::Union(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_id(item.hir_id());
            visitor.visit_variant_data(struct_definition);
        }
        ItemKind::Trait(.., ref generics, bounds, trait_item_refs) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_trait_item_ref, trait_item_refs);
        }
        ItemKind::TraitAlias(ref generics, bounds) => {
            visitor.visit_id(item.hir_id());
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
    }
}

pub fn walk_body<'v, V: Visitor<'v>>(visitor: &mut V, body: &'v Body<'v>) {
    walk_list!(visitor, visit_param, body.params);
    visitor.visit_expr(body.value);
}

pub fn walk_ident<'v, V: Visitor<'v>>(visitor: &mut V, ident: Ident) {
    visitor.visit_name(ident.name);
}

pub fn walk_mod<'v, V: Visitor<'v>>(visitor: &mut V, module: &'v Mod<'v>, mod_hir_id: HirId) {
    visitor.visit_id(mod_hir_id);
    for &item_id in module.item_ids {
        visitor.visit_nested_item(item_id);
    }
}

pub fn walk_foreign_item<'v, V: Visitor<'v>>(visitor: &mut V, foreign_item: &'v ForeignItem<'v>) {
    visitor.visit_id(foreign_item.hir_id());
    visitor.visit_ident(foreign_item.ident);

    match foreign_item.kind {
        ForeignItemKind::Fn(ref function_declaration, param_names, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_fn_decl(function_declaration);
            for &param_name in param_names {
                visitor.visit_ident(param_name);
            }
        }
        ForeignItemKind::Static(ref typ, _) => visitor.visit_ty(typ),
        ForeignItemKind::Type => (),
    }
}

pub fn walk_local<'v, V: Visitor<'v>>(visitor: &mut V, local: &'v Local<'v>) {
    // Intentionally visiting the expr first - the initialization expr
    // dominates the local's definition.
    walk_list!(visitor, visit_expr, &local.init);
    visitor.visit_id(local.hir_id);
    visitor.visit_pat(local.pat);
    if let Some(els) = local.els {
        visitor.visit_block(els);
    }
    walk_list!(visitor, visit_ty, &local.ty);
}

pub fn walk_block<'v, V: Visitor<'v>>(visitor: &mut V, block: &'v Block<'v>) {
    visitor.visit_id(block.hir_id);
    walk_list!(visitor, visit_stmt, block.stmts);
    walk_list!(visitor, visit_expr, &block.expr);
}

pub fn walk_stmt<'v, V: Visitor<'v>>(visitor: &mut V, statement: &'v Stmt<'v>) {
    visitor.visit_id(statement.hir_id);
    match statement.kind {
        StmtKind::Local(ref local) => visitor.visit_local(local),
        StmtKind::Item(item) => visitor.visit_nested_item(item),
        StmtKind::Expr(ref expression) | StmtKind::Semi(ref expression) => {
            visitor.visit_expr(expression)
        }
    }
}

pub fn walk_arm<'v, V: Visitor<'v>>(visitor: &mut V, arm: &'v Arm<'v>) {
    visitor.visit_id(arm.hir_id);
    visitor.visit_pat(arm.pat);
    if let Some(ref g) = arm.guard {
        match g {
            Guard::If(ref e) => visitor.visit_expr(e),
            Guard::IfLet(ref l) => {
                visitor.visit_let_expr(l);
            }
        }
    }
    visitor.visit_expr(arm.body);
}

pub fn walk_pat<'v, V: Visitor<'v>>(visitor: &mut V, pattern: &'v Pat<'v>) {
    visitor.visit_id(pattern.hir_id);
    match pattern.kind {
        PatKind::TupleStruct(ref qpath, children, _) => {
            visitor.visit_qpath(qpath, pattern.hir_id, pattern.span);
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref qpath) => {
            visitor.visit_qpath(qpath, pattern.hir_id, pattern.span);
        }
        PatKind::Struct(ref qpath, fields, _) => {
            visitor.visit_qpath(qpath, pattern.hir_id, pattern.span);
            walk_list!(visitor, visit_pat_field, fields);
        }
        PatKind::Or(pats) => walk_list!(visitor, visit_pat, pats),
        PatKind::Tuple(tuple_elements, _) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatKind::Box(ref subpattern) | PatKind::Ref(ref subpattern, _) => {
            visitor.visit_pat(subpattern)
        }
        PatKind::Binding(_, _hir_id, ident, ref optional_subpattern) => {
            visitor.visit_ident(ident);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => visitor.visit_expr(expression),
        PatKind::Range(ref lower_bound, ref upper_bound, _) => {
            walk_list!(visitor, visit_expr, lower_bound);
            walk_list!(visitor, visit_expr, upper_bound);
        }
        PatKind::Wild => (),
        PatKind::Slice(prepatterns, ref slice_pattern, postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
    }
}

pub fn walk_pat_field<'v, V: Visitor<'v>>(visitor: &mut V, field: &'v PatField<'v>) {
    visitor.visit_id(field.hir_id);
    visitor.visit_ident(field.ident);
    visitor.visit_pat(field.pat)
}

pub fn walk_array_len<'v, V: Visitor<'v>>(visitor: &mut V, len: &'v ArrayLen) {
    match len {
        &ArrayLen::Infer(hir_id, _span) => visitor.visit_id(hir_id),
        ArrayLen::Body(c) => visitor.visit_anon_const(c),
    }
}

pub fn walk_anon_const<'v, V: Visitor<'v>>(visitor: &mut V, constant: &'v AnonConst) {
    visitor.visit_id(constant.hir_id);
    visitor.visit_nested_body(constant.body);
}

pub fn walk_inline_const<'v, V: Visitor<'v>>(visitor: &mut V, constant: &'v ConstBlock) {
    visitor.visit_id(constant.hir_id);
    visitor.visit_nested_body(constant.body);
}

pub fn walk_expr<'v, V: Visitor<'v>>(visitor: &mut V, expression: &'v Expr<'v>) {
    visitor.visit_id(expression.hir_id);
    match expression.kind {
        ExprKind::Array(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::ConstBlock(ref const_block) => visitor.visit_inline_const(const_block),
        ExprKind::Repeat(ref element, ref count) => {
            visitor.visit_expr(element);
            visitor.visit_array_length(count)
        }
        ExprKind::Struct(ref qpath, fields, ref optional_base) => {
            visitor.visit_qpath(qpath, expression.hir_id, expression.span);
            walk_list!(visitor, visit_expr_field, fields);
            walk_list!(visitor, visit_expr, optional_base);
        }
        ExprKind::Tup(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(ref callee_expression, arguments) => {
            visitor.visit_expr(callee_expression);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(ref segment, receiver, arguments, _) => {
            visitor.visit_path_segment(segment);
            visitor.visit_expr(receiver);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, _, ref subexpression) | ExprKind::Unary(_, ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Cast(ref subexpression, ref typ) | ExprKind::Type(ref subexpression, ref typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::DropTemps(ref subexpression) => {
            visitor.visit_expr(subexpression);
        }
        ExprKind::Let(ref let_expr) => visitor.visit_let_expr(let_expr),
        ExprKind::If(ref cond, ref then, ref else_opt) => {
            visitor.visit_expr(cond);
            visitor.visit_expr(then);
            walk_list!(visitor, visit_expr, else_opt);
        }
        ExprKind::Loop(ref block, ref opt_label, _, _) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Match(ref subexpression, arms, _) => {
            visitor.visit_expr(subexpression);
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
            movability: _,
            constness: _,
        }) => {
            walk_list!(visitor, visit_generic_param, bound_generic_params);
            visitor.visit_fn(FnKind::Closure, fn_decl, body, expression.span, def_id)
        }
        ExprKind::Block(ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Assign(ref lhs, ref rhs, _) => {
            visitor.visit_expr(rhs);
            visitor.visit_expr(lhs)
        }
        ExprKind::AssignOp(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(right_expression);
            visitor.visit_expr(left_expression);
        }
        ExprKind::Field(ref subexpression, ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ident(ident);
        }
        ExprKind::Index(ref main_expression, ref index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprKind::Path(ref qpath) => {
            visitor.visit_qpath(qpath, expression.hir_id, expression.span);
        }
        ExprKind::Break(ref destination, ref opt_expr) => {
            walk_list!(visitor, visit_label, &destination.label);
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(ref destination) => {
            walk_list!(visitor, visit_label, &destination.label);
        }
        ExprKind::Ret(ref optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::InlineAsm(ref asm) => {
            visitor.visit_inline_asm(asm, expression.hir_id);
        }
        ExprKind::OffsetOf(ref container, ref fields) => {
            visitor.visit_ty(container);
            walk_list!(visitor, visit_ident, fields.iter().copied());
        }
        ExprKind::Yield(ref subexpression, _) => {
            visitor.visit_expr(subexpression);
        }
        ExprKind::Lit(_) | ExprKind::Err(_) => {}
    }
}

pub fn walk_let_expr<'v, V: Visitor<'v>>(visitor: &mut V, let_expr: &'v Let<'v>) {
    // match the visit order in walk_local
    visitor.visit_expr(let_expr.init);
    visitor.visit_id(let_expr.hir_id);
    visitor.visit_pat(let_expr.pat);
    walk_list!(visitor, visit_ty, let_expr.ty);
}

pub fn walk_expr_field<'v, V: Visitor<'v>>(visitor: &mut V, field: &'v ExprField<'v>) {
    visitor.visit_id(field.hir_id);
    visitor.visit_ident(field.ident);
    visitor.visit_expr(field.expr)
}

pub fn walk_ty<'v, V: Visitor<'v>>(visitor: &mut V, typ: &'v Ty<'v>) {
    visitor.visit_id(typ.hir_id);

    match typ.kind {
        TyKind::Slice(ref ty) => visitor.visit_ty(ty),
        TyKind::Ptr(ref mutable_type) => visitor.visit_ty(mutable_type.ty),
        TyKind::Ref(ref lifetime, ref mutable_type) => {
            visitor.visit_lifetime(lifetime);
            visitor.visit_ty(mutable_type.ty)
        }
        TyKind::Never => {}
        TyKind::Tup(tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(ref function_declaration) => {
            walk_list!(visitor, visit_generic_param, function_declaration.generic_params);
            visitor.visit_fn_decl(function_declaration.decl);
        }
        TyKind::Path(ref qpath) => {
            visitor.visit_qpath(qpath, typ.hir_id, typ.span);
        }
        TyKind::OpaqueDef(item_id, lifetimes, _in_trait) => {
            visitor.visit_nested_item(item_id);
            walk_list!(visitor, visit_generic_arg, lifetimes);
        }
        TyKind::Array(ref ty, ref length) => {
            visitor.visit_ty(ty);
            visitor.visit_array_length(length)
        }
        TyKind::TraitObject(bounds, ref lifetime, _syntax) => {
            for bound in bounds {
                visitor.visit_poly_trait_ref(bound);
            }
            visitor.visit_lifetime(lifetime);
        }
        TyKind::Typeof(ref expression) => visitor.visit_anon_const(expression),
        TyKind::Infer | TyKind::Err(_) => {}
    }
}

pub fn walk_generic_param<'v, V: Visitor<'v>>(visitor: &mut V, param: &'v GenericParam<'v>) {
    visitor.visit_id(param.hir_id);
    match param.name {
        ParamName::Plain(ident) => visitor.visit_ident(ident),
        ParamName::Error | ParamName::Fresh => {}
    }
    match param.kind {
        GenericParamKind::Lifetime { .. } => {}
        GenericParamKind::Type { ref default, .. } => walk_list!(visitor, visit_ty, default),
        GenericParamKind::Const { ref ty, ref default } => {
            visitor.visit_ty(ty);
            if let Some(ref default) = default {
                visitor.visit_const_param_default(param.hir_id, default);
            }
        }
    }
}

pub fn walk_const_param_default<'v, V: Visitor<'v>>(visitor: &mut V, ct: &'v AnonConst) {
    visitor.visit_anon_const(ct)
}

pub fn walk_generics<'v, V: Visitor<'v>>(visitor: &mut V, generics: &'v Generics<'v>) {
    walk_list!(visitor, visit_generic_param, generics.params);
    walk_list!(visitor, visit_where_predicate, generics.predicates);
}

pub fn walk_where_predicate<'v, V: Visitor<'v>>(
    visitor: &mut V,
    predicate: &'v WherePredicate<'v>,
) {
    match *predicate {
        WherePredicate::BoundPredicate(WhereBoundPredicate {
            hir_id,
            ref bounded_ty,
            bounds,
            bound_generic_params,
            origin: _,
            span: _,
        }) => {
            visitor.visit_id(hir_id);
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate {
            ref lifetime,
            bounds,
            span: _,
            in_where_clause: _,
        }) => {
            visitor.visit_lifetime(lifetime);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        WherePredicate::EqPredicate(WhereEqPredicate { ref lhs_ty, ref rhs_ty, span: _ }) => {
            visitor.visit_ty(lhs_ty);
            visitor.visit_ty(rhs_ty);
        }
    }
}

pub fn walk_fn_decl<'v, V: Visitor<'v>>(visitor: &mut V, function_declaration: &'v FnDecl<'v>) {
    for ty in function_declaration.inputs {
        visitor.visit_ty(ty)
    }
    visitor.visit_fn_ret_ty(&function_declaration.output)
}

pub fn walk_fn_ret_ty<'v, V: Visitor<'v>>(visitor: &mut V, ret_ty: &'v FnRetTy<'v>) {
    if let FnRetTy::Return(ref output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn<'v, V: Visitor<'v>>(
    visitor: &mut V,
    function_kind: FnKind<'v>,
    function_declaration: &'v FnDecl<'v>,
    body_id: BodyId,
    _: LocalDefId,
) {
    visitor.visit_fn_decl(function_declaration);
    walk_fn_kind(visitor, function_kind);
    visitor.visit_nested_body(body_id)
}

pub fn walk_fn_kind<'v, V: Visitor<'v>>(visitor: &mut V, function_kind: FnKind<'v>) {
    match function_kind {
        FnKind::ItemFn(_, generics, ..) => {
            visitor.visit_generics(generics);
        }
        FnKind::Closure | FnKind::Method(..) => {}
    }
}

pub fn walk_use<'v, V: Visitor<'v>>(visitor: &mut V, path: &'v UsePath<'v>, hir_id: HirId) {
    visitor.visit_id(hir_id);
    let UsePath { segments, ref res, span } = *path;
    for &res in res {
        visitor.visit_path(&Path { segments, res, span }, hir_id);
    }
}

pub fn walk_trait_item<'v, V: Visitor<'v>>(visitor: &mut V, trait_item: &'v TraitItem<'v>) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let TraitItem { ident, generics, ref defaultness, ref kind, span, owner_id: _ } = *trait_item;
    let hir_id = trait_item.hir_id();
    visitor.visit_ident(ident);
    visitor.visit_generics(&generics);
    visitor.visit_defaultness(&defaultness);
    visitor.visit_id(hir_id);
    match *kind {
        TraitItemKind::Const(ref ty, default) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_nested_body, default);
        }
        TraitItemKind::Fn(ref sig, TraitFn::Required(param_names)) => {
            visitor.visit_fn_decl(sig.decl);
            for &param_name in param_names {
                visitor.visit_ident(param_name);
            }
        }
        TraitItemKind::Fn(ref sig, TraitFn::Provided(body_id)) => {
            visitor.visit_fn(
                FnKind::Method(ident, sig),
                sig.decl,
                body_id,
                span,
                trait_item.owner_id.def_id,
            );
        }
        TraitItemKind::Type(bounds, ref default) => {
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
    }
}

pub fn walk_trait_item_ref<'v, V: Visitor<'v>>(visitor: &mut V, trait_item_ref: &'v TraitItemRef) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let TraitItemRef { id, ident, ref kind, span: _ } = *trait_item_ref;
    visitor.visit_nested_trait_item(id);
    visitor.visit_ident(ident);
    visitor.visit_associated_item_kind(kind);
}

pub fn walk_impl_item<'v, V: Visitor<'v>>(visitor: &mut V, impl_item: &'v ImplItem<'v>) {
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

    visitor.visit_ident(ident);
    visitor.visit_generics(generics);
    visitor.visit_defaultness(defaultness);
    visitor.visit_id(impl_item.hir_id());
    match *kind {
        ImplItemKind::Const(ref ty, body) => {
            visitor.visit_ty(ty);
            visitor.visit_nested_body(body);
        }
        ImplItemKind::Fn(ref sig, body_id) => {
            visitor.visit_fn(
                FnKind::Method(impl_item.ident, sig),
                sig.decl,
                body_id,
                impl_item.span,
                impl_item.owner_id.def_id,
            );
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_ty(ty);
        }
    }
}

pub fn walk_foreign_item_ref<'v, V: Visitor<'v>>(
    visitor: &mut V,
    foreign_item_ref: &'v ForeignItemRef,
) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ForeignItemRef { id, ident, span: _ } = *foreign_item_ref;
    visitor.visit_nested_foreign_item(id);
    visitor.visit_ident(ident);
}

pub fn walk_impl_item_ref<'v, V: Visitor<'v>>(visitor: &mut V, impl_item_ref: &'v ImplItemRef) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ImplItemRef { id, ident, ref kind, span: _, trait_item_def_id: _ } = *impl_item_ref;
    visitor.visit_nested_impl_item(id);
    visitor.visit_ident(ident);
    visitor.visit_associated_item_kind(kind);
}

pub fn walk_trait_ref<'v, V: Visitor<'v>>(visitor: &mut V, trait_ref: &'v TraitRef<'v>) {
    visitor.visit_id(trait_ref.hir_ref_id);
    visitor.visit_path(trait_ref.path, trait_ref.hir_ref_id)
}

pub fn walk_param_bound<'v, V: Visitor<'v>>(visitor: &mut V, bound: &'v GenericBound<'v>) {
    match *bound {
        GenericBound::Trait(ref typ, _modifier) => {
            visitor.visit_poly_trait_ref(typ);
        }
        GenericBound::LangItemTrait(_, _span, hir_id, args) => {
            visitor.visit_id(hir_id);
            visitor.visit_generic_args(args);
        }
        GenericBound::Outlives(ref lifetime) => visitor.visit_lifetime(lifetime),
    }
}

pub fn walk_poly_trait_ref<'v, V: Visitor<'v>>(visitor: &mut V, trait_ref: &'v PolyTraitRef<'v>) {
    walk_list!(visitor, visit_generic_param, trait_ref.bound_generic_params);
    visitor.visit_trait_ref(&trait_ref.trait_ref);
}

pub fn walk_struct_def<'v, V: Visitor<'v>>(
    visitor: &mut V,
    struct_definition: &'v VariantData<'v>,
) {
    walk_list!(visitor, visit_id, struct_definition.ctor_hir_id());
    walk_list!(visitor, visit_field_def, struct_definition.fields());
}

pub fn walk_field_def<'v, V: Visitor<'v>>(visitor: &mut V, field: &'v FieldDef<'v>) {
    visitor.visit_id(field.hir_id);
    visitor.visit_ident(field.ident);
    visitor.visit_ty(field.ty);
}

pub fn walk_enum_def<'v, V: Visitor<'v>>(
    visitor: &mut V,
    enum_definition: &'v EnumDef<'v>,
    item_id: HirId,
) {
    visitor.visit_id(item_id);
    walk_list!(visitor, visit_variant, enum_definition.variants);
}

pub fn walk_variant<'v, V: Visitor<'v>>(visitor: &mut V, variant: &'v Variant<'v>) {
    visitor.visit_ident(variant.ident);
    visitor.visit_id(variant.hir_id);
    visitor.visit_variant_data(&variant.data);
    walk_list!(visitor, visit_anon_const, &variant.disr_expr);
}

pub fn walk_label<'v, V: Visitor<'v>>(visitor: &mut V, label: &'v Label) {
    visitor.visit_ident(label.ident);
}

pub fn walk_inf<'v, V: Visitor<'v>>(visitor: &mut V, inf: &'v InferArg) {
    visitor.visit_id(inf.hir_id);
}

pub fn walk_generic_arg<'v, V: Visitor<'v>>(visitor: &mut V, generic_arg: &'v GenericArg<'v>) {
    match generic_arg {
        GenericArg::Lifetime(lt) => visitor.visit_lifetime(lt),
        GenericArg::Type(ty) => visitor.visit_ty(ty),
        GenericArg::Const(ct) => visitor.visit_anon_const(&ct.value),
        GenericArg::Infer(inf) => visitor.visit_infer(inf),
    }
}

pub fn walk_lifetime<'v, V: Visitor<'v>>(visitor: &mut V, lifetime: &'v Lifetime) {
    visitor.visit_id(lifetime.hir_id);
    visitor.visit_ident(lifetime.ident);
}

pub fn walk_qpath<'v, V: Visitor<'v>>(visitor: &mut V, qpath: &'v QPath<'v>, id: HirId) {
    match *qpath {
        QPath::Resolved(ref maybe_qself, ref path) => {
            walk_list!(visitor, visit_ty, maybe_qself);
            visitor.visit_path(path, id)
        }
        QPath::TypeRelative(ref qself, ref segment) => {
            visitor.visit_ty(qself);
            visitor.visit_path_segment(segment);
        }
        QPath::LangItem(..) => {}
    }
}

pub fn walk_path<'v, V: Visitor<'v>>(visitor: &mut V, path: &Path<'v>) {
    for segment in path.segments {
        visitor.visit_path_segment(segment);
    }
}

pub fn walk_path_segment<'v, V: Visitor<'v>>(visitor: &mut V, segment: &'v PathSegment<'v>) {
    visitor.visit_ident(segment.ident);
    visitor.visit_id(segment.hir_id);
    if let Some(ref args) = segment.args {
        visitor.visit_generic_args(args);
    }
}

pub fn walk_generic_args<'v, V: Visitor<'v>>(visitor: &mut V, generic_args: &'v GenericArgs<'v>) {
    walk_list!(visitor, visit_generic_arg, generic_args.args);
    walk_list!(visitor, visit_assoc_type_binding, generic_args.bindings);
}

pub fn walk_assoc_type_binding<'v, V: Visitor<'v>>(
    visitor: &mut V,
    type_binding: &'v TypeBinding<'v>,
) {
    visitor.visit_id(type_binding.hir_id);
    visitor.visit_ident(type_binding.ident);
    visitor.visit_generic_args(type_binding.gen_args);
    match type_binding.kind {
        TypeBindingKind::Equality { ref term } => match term {
            Term::Ty(ref ty) => visitor.visit_ty(ty),
            Term::Const(ref c) => visitor.visit_anon_const(c),
        },
        TypeBindingKind::Constraint { bounds } => walk_list!(visitor, visit_param_bound, bounds),
    }
}

pub fn walk_associated_item_kind<'v, V: Visitor<'v>>(_: &mut V, _: &'v AssocItemKind) {
    // No visitable content here: this fn exists so you can call it if
    // the right thing to do, should content be added in the future,
    // would be to walk it.
}

pub fn walk_defaultness<'v, V: Visitor<'v>>(_: &mut V, _: &'v Defaultness) {
    // No visitable content here: this fn exists so you can call it if
    // the right thing to do, should content be added in the future,
    // would be to walk it.
}

pub fn walk_inline_asm<'v, V: Visitor<'v>>(visitor: &mut V, asm: &'v InlineAsm<'v>, id: HirId) {
    for (op, op_sp) in asm.operands {
        match op {
            InlineAsmOperand::In { expr, .. } | InlineAsmOperand::InOut { expr, .. } => {
                visitor.visit_expr(expr)
            }
            InlineAsmOperand::Out { expr, .. } => {
                if let Some(expr) = expr {
                    visitor.visit_expr(expr);
                }
            }
            InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                visitor.visit_expr(in_expr);
                if let Some(out_expr) = out_expr {
                    visitor.visit_expr(out_expr);
                }
            }
            InlineAsmOperand::Const { anon_const, .. }
            | InlineAsmOperand::SymFn { anon_const, .. } => visitor.visit_anon_const(anon_const),
            InlineAsmOperand::SymStatic { path, .. } => visitor.visit_qpath(path, id, *op_sp),
        }
    }
}
