//! HIR walker for walking the contents of nodes.
//!
//! **For an overview of the visitor strategy, see the docs on the
//! `super::itemlikevisit::ItemLikeVisitor` trait.**
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

use syntax::ast::{Ident, Name, Attribute};
use syntax_pos::Span;
use crate::hir::*;
use crate::hir::map::Map;
use super::itemlikevisit::DeepVisitor;

#[derive(Copy, Clone)]
pub enum FnKind<'a> {
    /// `#[xxx] pub async/const/extern "Abi" fn foo()`
    ItemFn(Ident, &'a Generics, FnHeader, &'a Visibility, &'a [Attribute]),

    /// `fn foo(&self)`
    Method(Ident, &'a MethodSig, Option<&'a Visibility>, &'a [Attribute]),

    /// `|x, y| {}`
    Closure(&'a [Attribute]),
}

impl<'a> FnKind<'a> {
    pub fn attrs(&self) -> &'a [Attribute] {
        match *self {
            FnKind::ItemFn(.., attrs) => attrs,
            FnKind::Method(.., attrs) => attrs,
            FnKind::Closure(attrs) => attrs,
        }
    }

    pub fn header(&self) -> Option<&FnHeader> {
        match *self {
            FnKind::ItemFn(_, _, ref header, _, _) => Some(header),
            FnKind::Method(_, ref sig, _, _) => Some(&sig.header),
            FnKind::Closure(_) => None,
        }
    }
}

/// Specifies what nested things a visitor wants to visit. The most
/// common choice is `OnlyBodies`, which will cause the visitor to
/// visit fn bodies for fns that it encounters, but skip over nested
/// item-like things.
///
/// See the comments on `ItemLikeVisitor` for more details on the overall
/// visit strategy.
pub enum NestedVisitorMap<'this, 'tcx> {
    /// Do not visit any nested things. When you add a new
    /// "non-nested" thing, you will want to audit such uses to see if
    /// they remain valid.
    ///
    /// Use this if you are only walking some particular kind of tree
    /// (i.e., a type, or fn signature) and you don't want to thread a
    /// HIR map around.
    None,

    /// Do not visit nested item-like things, but visit nested things
    /// that are inside of an item-like.
    ///
    /// **This is the most common choice.** A very common pattern is
    /// to use `visit_all_item_likes()` as an outer loop,
    /// and to have the visitor that visits the contents of each item
    /// using this setting.
    OnlyBodies(&'this Map<'tcx>),

    /// Visits all nested things, including item-likes.
    ///
    /// **This is an unusual choice.** It is used when you want to
    /// process everything within their lexical context. Typically you
    /// kick off the visit by doing `walk_krate()`.
    All(&'this Map<'tcx>),
}

impl<'this, 'tcx> NestedVisitorMap<'this, 'tcx> {
    /// Returns the map to use for an "intra item-like" thing (if any).
    /// E.g., function body.
    pub fn intra(self) -> Option<&'this Map<'tcx>> {
        match self {
            NestedVisitorMap::None => None,
            NestedVisitorMap::OnlyBodies(map) => Some(map),
            NestedVisitorMap::All(map) => Some(map),
        }
    }

    /// Returns the map to use for an "item-like" thing (if any).
    /// E.g., item, impl-item.
    pub fn inter(self) -> Option<&'this Map<'tcx>> {
        match self {
            NestedVisitorMap::None => None,
            NestedVisitorMap::OnlyBodies(_) => None,
            NestedVisitorMap::All(map) => Some(map),
        }
    }
}

/// Each method of the Visitor trait is a hook to be potentially
/// overridden. Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_mod` method by default calls `intravisit::walk_mod`.
///
/// Note that this visitor does NOT visit nested items by default
/// (this is why the module is called `intravisit`, to distinguish it
/// from the AST's `visit` module, which acts differently). If you
/// simply want to visit all items in the crate in some order, you
/// should call `Crate::visit_all_items`. Otherwise, see the comment
/// on `visit_nested_item` for details on how to visit nested items.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method. (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<'v> : Sized {
    ///////////////////////////////////////////////////////////////////////////
    // Nested items.

    /// The default versions of the `visit_nested_XXX` routines invoke
    /// this method to get a map to use. By selecting an enum variant,
    /// you control which kinds of nested HIR are visited; see
    /// `NestedVisitorMap` for details. By "nested HIR", we are
    /// referring to bits of HIR that are not directly embedded within
    /// one another but rather indirectly, through a table in the
    /// crate. This is done to control dependencies during incremental
    /// compilation: the non-inline bits of HIR can be tracked and
    /// hashed separately.
    ///
    /// **If for some reason you want the nested behavior, but don't
    /// have a `Map` at your disposal:** then you should override the
    /// `visit_nested_XXX` methods, and override this method to
    /// `panic!()`. This way, if a new `visit_nested_XXX` variant is
    /// added in the future, we will see the panic in your code and
    /// fix it appropriately.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v>;

    /// Invoked when a nested item is encountered. By default does
    /// nothing unless you override `nested_visit_map` to return
    /// `Some(_)`, in which case it will walk the item. **You probably
    /// don't want to override this method** -- instead, override
    /// `nested_visit_map` or use the "shallow" or "deep" visit
    /// patterns described on `itemlikevisit::ItemLikeVisitor`. The only
    /// reason to override this method is if you want a nested pattern
    /// but cannot supply a `Map`; see `nested_visit_map` for advice.
    #[allow(unused_variables)]
    fn visit_nested_item(&mut self, id: ItemId) {
        let opt_item = self.nested_visit_map().inter().map(|map| map.expect_item(id.id));
        if let Some(item) = opt_item {
            self.visit_item(item);
        }
    }

    /// Like `visit_nested_item()`, but for trait items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    #[allow(unused_variables)]
    fn visit_nested_trait_item(&mut self, id: TraitItemId) {
        let opt_item = self.nested_visit_map().inter().map(|map| map.trait_item(id));
        if let Some(item) = opt_item {
            self.visit_trait_item(item);
        }
    }

    /// Like `visit_nested_item()`, but for impl items. See
    /// `visit_nested_item()` for advice on when to override this
    /// method.
    #[allow(unused_variables)]
    fn visit_nested_impl_item(&mut self, id: ImplItemId) {
        let opt_item = self.nested_visit_map().inter().map(|map| map.impl_item(id));
        if let Some(item) = opt_item {
            self.visit_impl_item(item);
        }
    }

    /// Invoked to visit the body of a function, method or closure. Like
    /// visit_nested_item, does nothing by default unless you override
    /// `nested_visit_map` to return `Some(_)`, in which case it will walk the
    /// body.
    fn visit_nested_body(&mut self, id: BodyId) {
        let opt_body = self.nested_visit_map().intra().map(|map| map.body(id));
        if let Some(body) = opt_body {
            self.visit_body(body);
        }
    }

    /// Visits the top-level item and (optionally) nested items / impl items. See
    /// `visit_nested_item` for details.
    fn visit_item(&mut self, i: &'v Item) {
        walk_item(self, i)
    }

    fn visit_body(&mut self, b: &'v Body) {
        walk_body(self, b);
    }

    /// When invoking `visit_all_item_likes()`, you need to supply an
    /// item-like visitor. This method converts a "intra-visit"
    /// visitor into an item-like visitor that walks the entire tree.
    /// If you use this, you probably don't want to process the
    /// contents of nested item-like things, since the outer loop will
    /// visit them as well.
    fn as_deep_visitor<'s>(&'s mut self) -> DeepVisitor<'s, Self> {
        DeepVisitor::new(self)
    }

    ///////////////////////////////////////////////////////////////////////////

    fn visit_id(&mut self, _hir_id: HirId) {
        // Nothing to do.
    }
    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, ident: Ident) {
        walk_ident(self, ident)
    }
    fn visit_mod(&mut self, m: &'v Mod, _s: Span, n: HirId) {
        walk_mod(self, m, n)
    }
    fn visit_foreign_item(&mut self, i: &'v ForeignItem) {
        walk_foreign_item(self, i)
    }
    fn visit_local(&mut self, l: &'v Local) {
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'v Block) {
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'v Stmt) {
        walk_stmt(self, s)
    }
    fn visit_arm(&mut self, a: &'v Arm) {
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'v Pat) {
        walk_pat(self, p)
    }
    fn visit_anon_const(&mut self, c: &'v AnonConst) {
        walk_anon_const(self, c)
    }
    fn visit_expr(&mut self, ex: &'v Expr) {
        walk_expr(self, ex)
    }
    fn visit_ty(&mut self, t: &'v Ty) {
        walk_ty(self, t)
    }
    fn visit_generic_param(&mut self, p: &'v GenericParam) {
        walk_generic_param(self, p)
    }
    fn visit_generics(&mut self, g: &'v Generics) {
        walk_generics(self, g)
    }
    fn visit_where_predicate(&mut self, predicate: &'v WherePredicate) {
        walk_where_predicate(self, predicate)
    }
    fn visit_fn_decl(&mut self, fd: &'v FnDecl) {
        walk_fn_decl(self, fd)
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: BodyId, s: Span, id: HirId) {
        walk_fn(self, fk, fd, b, s, id)
    }
    fn visit_use(&mut self, path: &'v Path, hir_id: HirId) {
        walk_use(self, path, hir_id)
    }
    fn visit_trait_item(&mut self, ti: &'v TraitItem) {
        walk_trait_item(self, ti)
    }
    fn visit_trait_item_ref(&mut self, ii: &'v TraitItemRef) {
        walk_trait_item_ref(self, ii)
    }
    fn visit_impl_item(&mut self, ii: &'v ImplItem) {
        walk_impl_item(self, ii)
    }
    fn visit_impl_item_ref(&mut self, ii: &'v ImplItemRef) {
        walk_impl_item_ref(self, ii)
    }
    fn visit_trait_ref(&mut self, t: &'v TraitRef) {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'v GenericBound) {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'v PolyTraitRef, m: TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self,
                          s: &'v VariantData,
                          _: Name,
                          _: &'v Generics,
                          _parent_id: HirId,
                          _: Span) {
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'v StructField) {
        walk_struct_field(self, s)
    }
    fn visit_enum_def(&mut self,
                      enum_definition: &'v EnumDef,
                      generics: &'v Generics,
                      item_id: HirId,
                      _: Span) {
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(&mut self, v: &'v Variant, g: &'v Generics, item_id: HirId) {
        walk_variant(self, v, g, item_id)
    }
    fn visit_label(&mut self, label: &'v Label) {
        walk_label(self, label)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'v GenericArg) {
        match generic_arg {
            GenericArg::Lifetime(lt) => self.visit_lifetime(lt),
            GenericArg::Type(ty) => self.visit_ty(ty),
            GenericArg::Const(ct) => self.visit_anon_const(&ct.value),
        }
    }
    fn visit_lifetime(&mut self, lifetime: &'v Lifetime) {
        walk_lifetime(self, lifetime)
    }
    fn visit_qpath(&mut self, qpath: &'v QPath, id: HirId, span: Span) {
        walk_qpath(self, qpath, id, span)
    }
    fn visit_path(&mut self, path: &'v Path, _id: HirId) {
        walk_path(self, path)
    }
    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'v PathSegment) {
        walk_path_segment(self, path_span, path_segment)
    }
    fn visit_generic_args(&mut self, path_span: Span, generic_args: &'v GenericArgs) {
        walk_generic_args(self, path_span, generic_args)
    }
    fn visit_assoc_type_binding(&mut self, type_binding: &'v TypeBinding) {
        walk_assoc_type_binding(self, type_binding)
    }
    fn visit_attribute(&mut self, _attr: &'v Attribute) {
    }
    fn visit_macro_def(&mut self, macro_def: &'v MacroDef) {
        walk_macro_def(self, macro_def)
    }
    fn visit_vis(&mut self, vis: &'v Visibility) {
        walk_vis(self, vis)
    }
    fn visit_associated_item_kind(&mut self, kind: &'v AssocItemKind) {
        walk_associated_item_kind(self, kind);
    }
    fn visit_defaultness(&mut self, defaultness: &'v Defaultness) {
        walk_defaultness(self, defaultness);
    }
}

/// Walks the contents of a crate. See also `Crate::visit_all_items`.
pub fn walk_crate<'v, V: Visitor<'v>>(visitor: &mut V, krate: &'v Crate) {
    visitor.visit_mod(&krate.module, krate.span, CRATE_HIR_ID);
    walk_list!(visitor, visit_attribute, &krate.attrs);
    walk_list!(visitor, visit_macro_def, &krate.exported_macros);
}

pub fn walk_macro_def<'v, V: Visitor<'v>>(visitor: &mut V, macro_def: &'v MacroDef) {
    visitor.visit_id(macro_def.hir_id);
    visitor.visit_name(macro_def.span, macro_def.name);
    walk_list!(visitor, visit_attribute, &macro_def.attrs);
}

pub fn walk_mod<'v, V: Visitor<'v>>(visitor: &mut V, module: &'v Mod, mod_hir_id: HirId) {
    visitor.visit_id(mod_hir_id);
    for &item_id in &module.item_ids {
        visitor.visit_nested_item(item_id);
    }
}

pub fn walk_body<'v, V: Visitor<'v>>(visitor: &mut V, body: &'v Body) {
    for argument in &body.arguments {
        visitor.visit_id(argument.hir_id);
        visitor.visit_pat(&argument.pat);
    }
    visitor.visit_expr(&body.value);
}

pub fn walk_local<'v, V: Visitor<'v>>(visitor: &mut V, local: &'v Local) {
    // Intentionally visiting the expr first - the initialization expr
    // dominates the local's definition.
    walk_list!(visitor, visit_expr, &local.init);
    walk_list!(visitor, visit_attribute, local.attrs.iter());
    visitor.visit_id(local.hir_id);
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
}

pub fn walk_ident<'v, V: Visitor<'v>>(visitor: &mut V, ident: Ident) {
    visitor.visit_name(ident.span, ident.name);
}

pub fn walk_label<'v, V: Visitor<'v>>(visitor: &mut V, label: &'v Label) {
    visitor.visit_ident(label.ident);
}

pub fn walk_lifetime<'v, V: Visitor<'v>>(visitor: &mut V, lifetime: &'v Lifetime) {
    visitor.visit_id(lifetime.hir_id);
    match lifetime.name {
        LifetimeName::Param(ParamName::Plain(ident)) => {
            visitor.visit_ident(ident);
        }
        LifetimeName::Param(ParamName::Fresh(_)) |
        LifetimeName::Param(ParamName::Error) |
        LifetimeName::Static |
        LifetimeName::Error |
        LifetimeName::Implicit |
        LifetimeName::Underscore => {}
    }
}

pub fn walk_poly_trait_ref<'v, V>(visitor: &mut V,
                                  trait_ref: &'v PolyTraitRef,
                                  _modifier: TraitBoundModifier)
    where V: Visitor<'v>
{
    walk_list!(visitor, visit_generic_param, &trait_ref.bound_generic_params);
    visitor.visit_trait_ref(&trait_ref.trait_ref);
}

pub fn walk_trait_ref<'v, V>(visitor: &mut V, trait_ref: &'v TraitRef)
    where V: Visitor<'v>
{
    visitor.visit_id(trait_ref.hir_ref_id);
    visitor.visit_path(&trait_ref.path, trait_ref.hir_ref_id)
}

pub fn walk_item<'v, V: Visitor<'v>>(visitor: &mut V, item: &'v Item) {
    visitor.visit_vis(&item.vis);
    visitor.visit_ident(item.ident);
    match item.node {
        ItemKind::ExternCrate(orig_name) => {
            visitor.visit_id(item.hir_id);
            if let Some(orig_name) = orig_name {
                visitor.visit_name(item.span, orig_name);
            }
        }
        ItemKind::Use(ref path, _) => {
            visitor.visit_use(path, item.hir_id);
        }
        ItemKind::Static(ref typ, _, body) |
        ItemKind::Const(ref typ, body) => {
            visitor.visit_id(item.hir_id);
            visitor.visit_ty(typ);
            visitor.visit_nested_body(body);
        }
        ItemKind::Fn(ref declaration, header, ref generics, body_id) => {
            visitor.visit_fn(FnKind::ItemFn(item.ident,
                                            generics,
                                            header,
                                            &item.vis,
                                            &item.attrs),
                             declaration,
                             body_id,
                             item.span,
                             item.hir_id)
        }
        ItemKind::Mod(ref module) => {
            // `visit_mod()` takes care of visiting the `Item`'s `HirId`.
            visitor.visit_mod(module, item.span, item.hir_id)
        }
        ItemKind::ForeignMod(ref foreign_module) => {
            visitor.visit_id(item.hir_id);
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemKind::GlobalAsm(_) => {
            visitor.visit_id(item.hir_id);
        }
        ItemKind::Ty(ref ty, ref generics) => {
            visitor.visit_id(item.hir_id);
            visitor.visit_ty(ty);
            visitor.visit_generics(generics)
        }
        ItemKind::Existential(ExistTy {
            ref generics,
            ref bounds,
            ..
        }) => {
            visitor.visit_id(item.hir_id);
            walk_generics(visitor, generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ItemKind::Enum(ref enum_definition, ref generics) => {
            visitor.visit_generics(generics);
            // `visit_enum_def()` takes care of visiting the `Item`'s `HirId`.
            visitor.visit_enum_def(enum_definition, generics, item.hir_id, item.span)
        }
        ItemKind::Impl(
            ..,
            ref generics,
            ref opt_trait_reference,
            ref typ,
            ref impl_item_refs
        ) => {
            visitor.visit_id(item.hir_id);
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_impl_item_ref, impl_item_refs);
        }
        ItemKind::Struct(ref struct_definition, ref generics) |
        ItemKind::Union(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_id(item.hir_id);
            visitor.visit_variant_data(struct_definition, item.ident.name, generics, item.hir_id,
                                       item.span);
        }
        ItemKind::Trait(.., ref generics, ref bounds, ref trait_item_refs) => {
            visitor.visit_id(item.hir_id);
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_trait_item_ref, trait_item_refs);
        }
        ItemKind::TraitAlias(ref generics, ref bounds) => {
            visitor.visit_id(item.hir_id);
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_use<'v, V: Visitor<'v>>(visitor: &mut V,
                                    path: &'v Path,
                                    hir_id: HirId) {
    visitor.visit_id(hir_id);
    visitor.visit_path(path, hir_id);
}

pub fn walk_enum_def<'v, V: Visitor<'v>>(visitor: &mut V,
                                         enum_definition: &'v EnumDef,
                                         generics: &'v Generics,
                                         item_id: HirId) {
    visitor.visit_id(item_id);
    walk_list!(visitor,
               visit_variant,
               &enum_definition.variants,
               generics,
               item_id);
}

pub fn walk_variant<'v, V: Visitor<'v>>(visitor: &mut V,
                                        variant: &'v Variant,
                                        generics: &'v Generics,
                                        parent_item_id: HirId) {
    visitor.visit_ident(variant.node.ident);
    visitor.visit_id(variant.node.id);
    visitor.visit_variant_data(&variant.node.data,
                               variant.node.ident.name,
                               generics,
                               parent_item_id,
                               variant.span);
    walk_list!(visitor, visit_anon_const, &variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.node.attrs);
}

pub fn walk_ty<'v, V: Visitor<'v>>(visitor: &mut V, typ: &'v Ty) {
    visitor.visit_id(typ.hir_id);

    match typ.node {
        TyKind::Slice(ref ty) => {
            visitor.visit_ty(ty)
        }
        TyKind::Ptr(ref mutable_type) => {
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Rptr(ref lifetime, ref mutable_type) => {
            visitor.visit_lifetime(lifetime);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Never => {},
        TyKind::Tup(ref tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(ref function_declaration) => {
            walk_list!(visitor, visit_generic_param, &function_declaration.generic_params);
            visitor.visit_fn_decl(&function_declaration.decl);
        }
        TyKind::Path(ref qpath) => {
            visitor.visit_qpath(qpath, typ.hir_id, typ.span);
        }
        TyKind::Def(item_id, ref lifetimes) => {
            visitor.visit_nested_item(item_id);
            walk_list!(visitor, visit_generic_arg, lifetimes);
        }
        TyKind::Array(ref ty, ref length) => {
            visitor.visit_ty(ty);
            visitor.visit_anon_const(length)
        }
        TyKind::TraitObject(ref bounds, ref lifetime) => {
            for bound in bounds {
                visitor.visit_poly_trait_ref(bound, TraitBoundModifier::None);
            }
            visitor.visit_lifetime(lifetime);
        }
        TyKind::Typeof(ref expression) => {
            visitor.visit_anon_const(expression)
        }
        TyKind::CVarArgs(ref lt) => {
            visitor.visit_lifetime(lt)
        }
        TyKind::Infer | TyKind::Err => {}
    }
}

pub fn walk_qpath<'v, V: Visitor<'v>>(visitor: &mut V, qpath: &'v QPath, id: HirId, span: Span) {
    match *qpath {
        QPath::Resolved(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(qself);
            }
            visitor.visit_path(path, id)
        }
        QPath::TypeRelative(ref qself, ref segment) => {
            visitor.visit_ty(qself);
            visitor.visit_path_segment(span, segment);
        }
    }
}

pub fn walk_path<'v, V: Visitor<'v>>(visitor: &mut V, path: &'v Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_path_segment<'v, V: Visitor<'v>>(visitor: &mut V,
                                             path_span: Span,
                                             segment: &'v PathSegment) {
    visitor.visit_ident(segment.ident);
    if let Some(id) = segment.hir_id {
        visitor.visit_id(id);
    }
    if let Some(ref args) = segment.args {
        visitor.visit_generic_args(path_span, args);
    }
}

pub fn walk_generic_args<'v, V: Visitor<'v>>(visitor: &mut V,
                                             _path_span: Span,
                                             generic_args: &'v GenericArgs) {
    walk_list!(visitor, visit_generic_arg, &generic_args.args);
    walk_list!(visitor, visit_assoc_type_binding, &generic_args.bindings);
}

pub fn walk_assoc_type_binding<'v, V: Visitor<'v>>(visitor: &mut V,
                                                   type_binding: &'v TypeBinding) {
    visitor.visit_id(type_binding.hir_id);
    visitor.visit_ident(type_binding.ident);
    match type_binding.kind {
        TypeBindingKind::Equality { ref ty } => {
            visitor.visit_ty(ty);
        }
        TypeBindingKind::Constraint { ref bounds } => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
    }
}

pub fn walk_pat<'v, V: Visitor<'v>>(visitor: &mut V, pattern: &'v Pat) {
    visitor.visit_id(pattern.hir_id);
    match pattern.node {
        PatKind::TupleStruct(ref qpath, ref children, _) => {
            visitor.visit_qpath(qpath, pattern.hir_id, pattern.span);
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref qpath) => {
            visitor.visit_qpath(qpath, pattern.hir_id, pattern.span);
        }
        PatKind::Struct(ref qpath, ref fields, _) => {
            visitor.visit_qpath(qpath, pattern.hir_id, pattern.span);
            for field in fields {
                visitor.visit_id(field.node.hir_id);
                visitor.visit_ident(field.node.ident);
                visitor.visit_pat(&field.node.pat)
            }
        }
        PatKind::Tuple(ref tuple_elements, _) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatKind::Box(ref subpattern) |
        PatKind::Ref(ref subpattern, _) => {
            visitor.visit_pat(subpattern)
        }
        PatKind::Binding(_, _hir_id, ident, ref optional_subpattern) => {
            visitor.visit_ident(ident);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => visitor.visit_expr(expression),
        PatKind::Range(ref lower_bound, ref upper_bound, _) => {
            visitor.visit_expr(lower_bound);
            visitor.visit_expr(upper_bound)
        }
        PatKind::Wild => (),
        PatKind::Slice(ref prepatterns, ref slice_pattern, ref postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
    }
}

pub fn walk_foreign_item<'v, V: Visitor<'v>>(visitor: &mut V, foreign_item: &'v ForeignItem) {
    visitor.visit_id(foreign_item.hir_id);
    visitor.visit_vis(&foreign_item.vis);
    visitor.visit_ident(foreign_item.ident);

    match foreign_item.node {
        ForeignItemKind::Fn(ref function_declaration, ref param_names, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_fn_decl(function_declaration);
            for &param_name in param_names {
                visitor.visit_ident(param_name);
            }
        }
        ForeignItemKind::Static(ref typ, _) => visitor.visit_ty(typ),
        ForeignItemKind::Type => (),
    }

    walk_list!(visitor, visit_attribute, &foreign_item.attrs);
}

pub fn walk_param_bound<'v, V: Visitor<'v>>(visitor: &mut V, bound: &'v GenericBound) {
    match *bound {
        GenericBound::Trait(ref typ, modifier) => {
            visitor.visit_poly_trait_ref(typ, modifier);
        }
        GenericBound::Outlives(ref lifetime) => visitor.visit_lifetime(lifetime),
    }
}

pub fn walk_generic_param<'v, V: Visitor<'v>>(visitor: &mut V, param: &'v GenericParam) {
    visitor.visit_id(param.hir_id);
    walk_list!(visitor, visit_attribute, &param.attrs);
    match param.name {
        ParamName::Plain(ident) => visitor.visit_ident(ident),
        ParamName::Error | ParamName::Fresh(_) => {}
    }
    match param.kind {
        GenericParamKind::Lifetime { .. } => {}
        GenericParamKind::Type { ref default, .. } => walk_list!(visitor, visit_ty, default),
        GenericParamKind::Const { ref ty } => visitor.visit_ty(ty),
    }
    walk_list!(visitor, visit_param_bound, &param.bounds);
}

pub fn walk_generics<'v, V: Visitor<'v>>(visitor: &mut V, generics: &'v Generics) {
    walk_list!(visitor, visit_generic_param, &generics.params);
    walk_list!(visitor, visit_where_predicate, &generics.where_clause.predicates);
}

pub fn walk_where_predicate<'v, V: Visitor<'v>>(
    visitor: &mut V,
    predicate: &'v WherePredicate)
{
    match predicate {
        &WherePredicate::BoundPredicate(WhereBoundPredicate{ref bounded_ty,
                                                            ref bounds,
                                                            ref bound_generic_params,
                                                            ..}) => {
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        &WherePredicate::RegionPredicate(WhereRegionPredicate{ref lifetime,
                                                              ref bounds,
                                                              ..}) => {
            visitor.visit_lifetime(lifetime);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        &WherePredicate::EqPredicate(WhereEqPredicate{hir_id,
                                                      ref lhs_ty,
                                                      ref rhs_ty,
                                                      ..}) => {
            visitor.visit_id(hir_id);
            visitor.visit_ty(lhs_ty);
            visitor.visit_ty(rhs_ty);
        }
    }
}

pub fn walk_fn_ret_ty<'v, V: Visitor<'v>>(visitor: &mut V, ret_ty: &'v FunctionRetTy) {
    if let Return(ref output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<'v, V: Visitor<'v>>(visitor: &mut V, function_declaration: &'v FnDecl) {
    for ty in &function_declaration.inputs {
        visitor.visit_ty(ty)
    }
    walk_fn_ret_ty(visitor, &function_declaration.output)
}

pub fn walk_fn_kind<'v, V: Visitor<'v>>(visitor: &mut V, function_kind: FnKind<'v>) {
    match function_kind {
        FnKind::ItemFn(_, generics, ..) => {
            visitor.visit_generics(generics);
        }
        FnKind::Method(..) |
        FnKind::Closure(_) => {}
    }
}

pub fn walk_fn<'v, V: Visitor<'v>>(visitor: &mut V,
                                   function_kind: FnKind<'v>,
                                   function_declaration: &'v FnDecl,
                                   body_id: BodyId,
                                   _span: Span,
                                   id: HirId) {
    visitor.visit_id(id);
    visitor.visit_fn_decl(function_declaration);
    walk_fn_kind(visitor, function_kind);
    visitor.visit_nested_body(body_id)
}

pub fn walk_trait_item<'v, V: Visitor<'v>>(visitor: &mut V, trait_item: &'v TraitItem) {
    visitor.visit_ident(trait_item.ident);
    walk_list!(visitor, visit_attribute, &trait_item.attrs);
    visitor.visit_generics(&trait_item.generics);
    match trait_item.node {
        TraitItemKind::Const(ref ty, default) => {
            visitor.visit_id(trait_item.hir_id);
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_nested_body, default);
        }
        TraitItemKind::Method(ref sig, TraitMethod::Required(ref param_names)) => {
            visitor.visit_id(trait_item.hir_id);
            visitor.visit_fn_decl(&sig.decl);
            for &param_name in param_names {
                visitor.visit_ident(param_name);
            }
        }
        TraitItemKind::Method(ref sig, TraitMethod::Provided(body_id)) => {
            visitor.visit_fn(FnKind::Method(trait_item.ident,
                                            sig,
                                            None,
                                            &trait_item.attrs),
                             &sig.decl,
                             body_id,
                             trait_item.span,
                             trait_item.hir_id);
        }
        TraitItemKind::Type(ref bounds, ref default) => {
            visitor.visit_id(trait_item.hir_id);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
    }
}

pub fn walk_trait_item_ref<'v, V: Visitor<'v>>(visitor: &mut V, trait_item_ref: &'v TraitItemRef) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let TraitItemRef { id, ident, ref kind, span: _, ref defaultness } = *trait_item_ref;
    visitor.visit_nested_trait_item(id);
    visitor.visit_ident(ident);
    visitor.visit_associated_item_kind(kind);
    visitor.visit_defaultness(defaultness);
}

pub fn walk_impl_item<'v, V: Visitor<'v>>(visitor: &mut V, impl_item: &'v ImplItem) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ImplItem {
        hir_id: _,
        ident,
        ref vis,
        ref defaultness,
        ref attrs,
        ref generics,
        ref node,
        span: _,
    } = *impl_item;

    visitor.visit_ident(ident);
    visitor.visit_vis(vis);
    visitor.visit_defaultness(defaultness);
    walk_list!(visitor, visit_attribute, attrs);
    visitor.visit_generics(generics);
    match *node {
        ImplItemKind::Const(ref ty, body) => {
            visitor.visit_id(impl_item.hir_id);
            visitor.visit_ty(ty);
            visitor.visit_nested_body(body);
        }
        ImplItemKind::Method(ref sig, body_id) => {
            visitor.visit_fn(FnKind::Method(impl_item.ident,
                                            sig,
                                            Some(&impl_item.vis),
                                            &impl_item.attrs),
                             &sig.decl,
                             body_id,
                             impl_item.span,
                             impl_item.hir_id);
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_id(impl_item.hir_id);
            visitor.visit_ty(ty);
        }
        ImplItemKind::Existential(ref bounds) => {
            visitor.visit_id(impl_item.hir_id);
            walk_list!(visitor, visit_param_bound, bounds);
        }
    }
}

pub fn walk_impl_item_ref<'v, V: Visitor<'v>>(visitor: &mut V, impl_item_ref: &'v ImplItemRef) {
    // N.B., deliberately force a compilation error if/when new fields are added.
    let ImplItemRef { id, ident, ref kind, span: _, ref vis, ref defaultness } = *impl_item_ref;
    visitor.visit_nested_impl_item(id);
    visitor.visit_ident(ident);
    visitor.visit_associated_item_kind(kind);
    visitor.visit_vis(vis);
    visitor.visit_defaultness(defaultness);
}

pub fn walk_struct_def<'v, V: Visitor<'v>>(visitor: &mut V, struct_definition: &'v VariantData) {
    if let Some(ctor_hir_id) = struct_definition.ctor_hir_id() {
        visitor.visit_id(ctor_hir_id);
    }
    walk_list!(visitor, visit_struct_field, struct_definition.fields());
}

pub fn walk_struct_field<'v, V: Visitor<'v>>(visitor: &mut V, struct_field: &'v StructField) {
    visitor.visit_id(struct_field.hir_id);
    visitor.visit_vis(&struct_field.vis);
    visitor.visit_ident(struct_field.ident);
    visitor.visit_ty(&struct_field.ty);
    walk_list!(visitor, visit_attribute, &struct_field.attrs);
}

pub fn walk_block<'v, V: Visitor<'v>>(visitor: &mut V, block: &'v Block) {
    visitor.visit_id(block.hir_id);
    walk_list!(visitor, visit_stmt, &block.stmts);
    walk_list!(visitor, visit_expr, &block.expr);
}

pub fn walk_stmt<'v, V: Visitor<'v>>(visitor: &mut V, statement: &'v Stmt) {
    visitor.visit_id(statement.hir_id);
    match statement.node {
        StmtKind::Local(ref local) => visitor.visit_local(local),
        StmtKind::Item(item) => visitor.visit_nested_item(item),
        StmtKind::Expr(ref expression) |
        StmtKind::Semi(ref expression) => {
            visitor.visit_expr(expression)
        }
    }
}

pub fn walk_anon_const<'v, V: Visitor<'v>>(visitor: &mut V, constant: &'v AnonConst) {
    visitor.visit_id(constant.hir_id);
    visitor.visit_nested_body(constant.body);
}

pub fn walk_expr<'v, V: Visitor<'v>>(visitor: &mut V, expression: &'v Expr) {
    visitor.visit_id(expression.hir_id);
    walk_list!(visitor, visit_attribute, expression.attrs.iter());
    match expression.node {
        ExprKind::Box(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Array(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Repeat(ref element, ref count) => {
            visitor.visit_expr(element);
            visitor.visit_anon_const(count)
        }
        ExprKind::Struct(ref qpath, ref fields, ref optional_base) => {
            visitor.visit_qpath(qpath, expression.hir_id, expression.span);
            for field in fields {
                visitor.visit_id(field.hir_id);
                visitor.visit_ident(field.ident);
                visitor.visit_expr(&field.expr)
            }
            walk_list!(visitor, visit_expr, optional_base);
        }
        ExprKind::Tup(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(ref callee_expression, ref arguments) => {
            visitor.visit_expr(callee_expression);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(ref segment, _, ref arguments) => {
            visitor.visit_path_segment(expression.span, segment);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, ref subexpression) | ExprKind::Unary(_, ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Cast(ref subexpression, ref typ) | ExprKind::Type(ref subexpression, ref typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::DropTemps(ref subexpression) => {
            visitor.visit_expr(subexpression);
        }
        ExprKind::While(ref subexpression, ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::Loop(ref block, ref opt_label, _) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Match(ref subexpression, ref arms, _) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(_, ref function_declaration, body, _fn_decl_span, _gen) => {
            visitor.visit_fn(FnKind::Closure(&expression.attrs),
                             function_declaration,
                             body,
                             expression.span,
                             expression.hir_id)
        }
        ExprKind::Block(ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Assign(ref left_hand_expression, ref right_hand_expression) => {
            visitor.visit_expr(right_hand_expression);
            visitor.visit_expr(left_hand_expression)
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
            if let Some(ref label) = destination.label {
                visitor.visit_label(label);
            }
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(ref destination) => {
            if let Some(ref label) = destination.label {
                visitor.visit_label(label);
            }
        }
        ExprKind::Ret(ref optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::InlineAsm(_, ref outputs, ref inputs) => {
            for expr in outputs.iter().chain(inputs.iter()) {
                visitor.visit_expr(expr)
            }
        }
        ExprKind::Yield(ref subexpression, _) => {
            visitor.visit_expr(subexpression);
        }
        ExprKind::Lit(_) | ExprKind::Err => {}
    }
}

pub fn walk_arm<'v, V: Visitor<'v>>(visitor: &mut V, arm: &'v Arm) {
    visitor.visit_id(arm.hir_id);
    walk_list!(visitor, visit_pat, &arm.pats);
    if let Some(ref g) = arm.guard {
        match g {
            Guard::If(ref e) => visitor.visit_expr(e),
        }
    }
    visitor.visit_expr(&arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
}

pub fn walk_vis<'v, V: Visitor<'v>>(visitor: &mut V, vis: &'v Visibility) {
    if let VisibilityKind::Restricted { ref path, hir_id } = vis.node {
        visitor.visit_id(hir_id);
        visitor.visit_path(path, hir_id)
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
