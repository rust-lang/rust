// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! HIR walker for walking the contents of nodes.
//!
//! **For an overview of the visitor strategy, see the docs on the
//! `super::itemlikevisit::ItemLikeVisitor` trait.**
//!
//! If you have decided to use this visitor, here are some general
//! notes on how to do it:
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
//! the body of a function in "execution order" (more concretely,
//! reverse post-order with respect to the CFG implied by the AST),
//! meaning that if AST node A may execute before AST node B, then A
//! is visited first.  The borrow checker in particular relies on this
//! property.

use syntax::abi::Abi;
use syntax::ast::{NodeId, CRATE_NODE_ID, Name, Attribute};
use syntax::codemap::Spanned;
use syntax_pos::Span;
use hir::*;
use hir::def::Def;
use hir::map::Map;
use super::itemlikevisit::DeepVisitor;

use std::cmp;
use std::u32;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FnKind<'a> {
    /// fn foo() or extern "Abi" fn foo()
    ItemFn(Name, &'a Generics, Unsafety, Constness, Abi, &'a Visibility, &'a [Attribute]),

    /// fn foo(&self)
    Method(Name, &'a MethodSig, Option<&'a Visibility>, &'a [Attribute]),

    /// |x, y| {}
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
}

/// Specifies what nested things a visitor wants to visit. The most
/// common choice is `OnlyBodies`, which will cause the visitor to
/// visit fn bodies for fns that it encounters, but skip over nested
/// item-like things.
///
/// See the comments on `ItemLikeVisitor` for more details on the overall
/// visit strategy.
pub enum NestedVisitorMap<'this, 'tcx: 'this> {
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
    /// **This is the most common choice.** A very commmon pattern is
    /// to use `tcx.visit_all_item_likes_in_krate()` as an outer loop,
    /// and to have the visitor that visits the contents of each item
    /// using this setting.
    OnlyBodies(&'this Map<'tcx>),

    /// Visit all nested things, including item-likes.
    ///
    /// **This is an unusual choice.** It is used when you want to
    /// process everything within their lexical context. Typically you
    /// kick off the visit by doing `walk_krate()`.
    All(&'this Map<'tcx>),
}

impl<'this, 'tcx> NestedVisitorMap<'this, 'tcx> {
    /// Returns the map to use for an "intra item-like" thing (if any).
    /// e.g., function body.
    pub fn intra(self) -> Option<&'this Map<'tcx>> {
        match self {
            NestedVisitorMap::None => None,
            NestedVisitorMap::OnlyBodies(map) => Some(map),
            NestedVisitorMap::All(map) => Some(map),
        }
    }

    /// Returns the map to use for an "item-like" thing (if any).
    /// e.g., item, impl-item.
    pub fn inter(self) -> Option<&'this Map<'tcx>> {
        match self {
            NestedVisitorMap::None => None,
            NestedVisitorMap::OnlyBodies(_) => None,
            NestedVisitorMap::All(map) => Some(map),
        }
    }
}

/// Each method of the Visitor trait is a hook to be potentially
/// overridden.  Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g. the `visit_mod` method by default calls `intravisit::walk_mod`.
///
/// Note that this visitor does NOT visit nested items by default
/// (this is why the module is called `intravisit`, to distinguish it
/// from the AST's `visit` module, which acts differently). If you
/// simply want to visit all items in the crate in some order, you
/// should call `Crate::visit_all_items`. Otherwise, see the comment
/// on `visit_nested_item` for details on how to visit nested items.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method.  (And you also need
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
    /// have a `Map` are your disposal:** then you should override the
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

    /// Visit the top-level item and (optionally) nested items / impl items. See
    /// `visit_nested_item` for details.
    fn visit_item(&mut self, i: &'v Item) {
        walk_item(self, i)
    }

    fn visit_body(&mut self, b: &'v Body) {
        walk_body(self, b);
    }

    /// When invoking `visit_all_item_likes()`, you need to supply an
    /// item-like visitor.  This method converts a "intra-visit"
    /// visitor into an item-like visitor that walks the entire tree.
    /// If you use this, you probably don't want to process the
    /// contents of nested item-like things, since the outer loop will
    /// visit them as well.
    fn as_deep_visitor<'s>(&'s mut self) -> DeepVisitor<'s, Self> {
        DeepVisitor::new(self)
    }

    ///////////////////////////////////////////////////////////////////////////

    fn visit_id(&mut self, _node_id: NodeId) {
        // Nothing to do.
    }
    fn visit_def_mention(&mut self, _def: Def) {
        // Nothing to do.
    }
    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_mod(&mut self, m: &'v Mod, _s: Span, n: NodeId) {
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
    fn visit_decl(&mut self, d: &'v Decl) {
        walk_decl(self, d)
    }
    fn visit_expr(&mut self, ex: &'v Expr) {
        walk_expr(self, ex)
    }
    fn visit_ty(&mut self, t: &'v Ty) {
        walk_ty(self, t)
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
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: BodyId, s: Span, id: NodeId) {
        walk_fn(self, fk, fd, b, s, id)
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
    fn visit_ty_param_bound(&mut self, bounds: &'v TyParamBound) {
        walk_ty_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'v PolyTraitRef, m: &'v TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self,
                          s: &'v VariantData,
                          _: Name,
                          _: &'v Generics,
                          _parent_id: NodeId,
                          _: Span) {
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'v StructField) {
        walk_struct_field(self, s)
    }
    fn visit_enum_def(&mut self,
                      enum_definition: &'v EnumDef,
                      generics: &'v Generics,
                      item_id: NodeId,
                      _: Span) {
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(&mut self, v: &'v Variant, g: &'v Generics, item_id: NodeId) {
        walk_variant(self, v, g, item_id)
    }
    fn visit_lifetime(&mut self, lifetime: &'v Lifetime) {
        walk_lifetime(self, lifetime)
    }
    fn visit_lifetime_def(&mut self, lifetime: &'v LifetimeDef) {
        walk_lifetime_def(self, lifetime)
    }
    fn visit_qpath(&mut self, qpath: &'v QPath, id: NodeId, span: Span) {
        walk_qpath(self, qpath, id, span)
    }
    fn visit_path(&mut self, path: &'v Path, _id: NodeId) {
        walk_path(self, path)
    }
    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'v PathSegment) {
        walk_path_segment(self, path_span, path_segment)
    }
    fn visit_path_parameters(&mut self, path_span: Span, path_parameters: &'v PathParameters) {
        walk_path_parameters(self, path_span, path_parameters)
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
    fn visit_associated_item_kind(&mut self, kind: &'v AssociatedItemKind) {
        walk_associated_item_kind(self, kind);
    }
    fn visit_defaultness(&mut self, defaultness: &'v Defaultness) {
        walk_defaultness(self, defaultness);
    }
}

pub fn walk_opt_name<'v, V: Visitor<'v>>(visitor: &mut V, span: Span, opt_name: Option<Name>) {
    if let Some(name) = opt_name {
        visitor.visit_name(span, name);
    }
}

pub fn walk_opt_sp_name<'v, V: Visitor<'v>>(visitor: &mut V, opt_sp_name: &Option<Spanned<Name>>) {
    if let Some(ref sp_name) = *opt_sp_name {
        visitor.visit_name(sp_name.span, sp_name.node);
    }
}

/// Walks the contents of a crate. See also `Crate::visit_all_items`.
pub fn walk_crate<'v, V: Visitor<'v>>(visitor: &mut V, krate: &'v Crate) {
    visitor.visit_mod(&krate.module, krate.span, CRATE_NODE_ID);
    walk_list!(visitor, visit_attribute, &krate.attrs);
    walk_list!(visitor, visit_macro_def, &krate.exported_macros);
}

pub fn walk_macro_def<'v, V: Visitor<'v>>(visitor: &mut V, macro_def: &'v MacroDef) {
    visitor.visit_id(macro_def.id);
    visitor.visit_name(macro_def.span, macro_def.name);
    walk_list!(visitor, visit_attribute, &macro_def.attrs);
}

pub fn walk_mod<'v, V: Visitor<'v>>(visitor: &mut V, module: &'v Mod, mod_node_id: NodeId) {
    visitor.visit_id(mod_node_id);
    for &item_id in &module.item_ids {
        visitor.visit_nested_item(item_id);
    }
}

pub fn walk_body<'v, V: Visitor<'v>>(visitor: &mut V, body: &'v Body) {
    for argument in &body.arguments {
        visitor.visit_id(argument.id);
        visitor.visit_pat(&argument.pat);
    }
    visitor.visit_expr(&body.value);
}

pub fn walk_local<'v, V: Visitor<'v>>(visitor: &mut V, local: &'v Local) {
    visitor.visit_id(local.id);
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
    walk_list!(visitor, visit_expr, &local.init);
}

pub fn walk_lifetime<'v, V: Visitor<'v>>(visitor: &mut V, lifetime: &'v Lifetime) {
    visitor.visit_id(lifetime.id);
    visitor.visit_name(lifetime.span, lifetime.name);
}

pub fn walk_lifetime_def<'v, V: Visitor<'v>>(visitor: &mut V, lifetime_def: &'v LifetimeDef) {
    visitor.visit_lifetime(&lifetime_def.lifetime);
    walk_list!(visitor, visit_lifetime, &lifetime_def.bounds);
}

pub fn walk_poly_trait_ref<'v, V>(visitor: &mut V,
                                  trait_ref: &'v PolyTraitRef,
                                  _modifier: &'v TraitBoundModifier)
    where V: Visitor<'v>
{
    walk_list!(visitor, visit_lifetime_def, &trait_ref.bound_lifetimes);
    visitor.visit_trait_ref(&trait_ref.trait_ref);
}

pub fn walk_trait_ref<'v, V>(visitor: &mut V, trait_ref: &'v TraitRef)
    where V: Visitor<'v>
{
    visitor.visit_id(trait_ref.ref_id);
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<'v, V: Visitor<'v>>(visitor: &mut V, item: &'v Item) {
    visitor.visit_vis(&item.vis);
    visitor.visit_name(item.span, item.name);
    match item.node {
        ItemExternCrate(opt_name) => {
            visitor.visit_id(item.id);
            walk_opt_name(visitor, item.span, opt_name)
        }
        ItemUse(ref path, _) => {
            visitor.visit_id(item.id);
            visitor.visit_path(path, item.id);
        }
        ItemStatic(ref typ, _, body) |
        ItemConst(ref typ, body) => {
            visitor.visit_id(item.id);
            visitor.visit_ty(typ);
            visitor.visit_nested_body(body);
        }
        ItemFn(ref declaration, unsafety, constness, abi, ref generics, body_id) => {
            visitor.visit_fn(FnKind::ItemFn(item.name,
                                            generics,
                                            unsafety,
                                            constness,
                                            abi,
                                            &item.vis,
                                            &item.attrs),
                             declaration,
                             body_id,
                             item.span,
                             item.id)
        }
        ItemMod(ref module) => {
            // visit_mod() takes care of visiting the Item's NodeId
            visitor.visit_mod(module, item.span, item.id)
        }
        ItemForeignMod(ref foreign_module) => {
            visitor.visit_id(item.id);
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemTy(ref typ, ref type_parameters) => {
            visitor.visit_id(item.id);
            visitor.visit_ty(typ);
            visitor.visit_generics(type_parameters)
        }
        ItemEnum(ref enum_definition, ref type_parameters) => {
            visitor.visit_generics(type_parameters);
            // visit_enum_def() takes care of visiting the Item's NodeId
            visitor.visit_enum_def(enum_definition, type_parameters, item.id, item.span)
        }
        ItemDefaultImpl(_, ref trait_ref) => {
            visitor.visit_id(item.id);
            visitor.visit_trait_ref(trait_ref)
        }
        ItemImpl(.., ref type_parameters, ref opt_trait_reference, ref typ, ref impl_item_refs) => {
            visitor.visit_id(item.id);
            visitor.visit_generics(type_parameters);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_impl_item_ref, impl_item_refs);
        }
        ItemStruct(ref struct_definition, ref generics) |
        ItemUnion(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_id(item.id);
            visitor.visit_variant_data(struct_definition, item.name, generics, item.id, item.span);
        }
        ItemTrait(_, ref generics, ref bounds, ref trait_item_refs) => {
            visitor.visit_id(item.id);
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_trait_item_ref, trait_item_refs);
        }
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'v, V: Visitor<'v>>(visitor: &mut V,
                                         enum_definition: &'v EnumDef,
                                         generics: &'v Generics,
                                         item_id: NodeId) {
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
                                        parent_item_id: NodeId) {
    visitor.visit_name(variant.span, variant.node.name);
    visitor.visit_variant_data(&variant.node.data,
                               variant.node.name,
                               generics,
                               parent_item_id,
                               variant.span);
    walk_list!(visitor, visit_nested_body, variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.node.attrs);
}

pub fn walk_ty<'v, V: Visitor<'v>>(visitor: &mut V, typ: &'v Ty) {
    visitor.visit_id(typ.id);

    match typ.node {
        TySlice(ref ty) => {
            visitor.visit_ty(ty)
        }
        TyPtr(ref mutable_type) => {
            visitor.visit_ty(&mutable_type.ty)
        }
        TyRptr(ref opt_lifetime, ref mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyNever => {},
        TyTup(ref tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyBareFn(ref function_declaration) => {
            visitor.visit_fn_decl(&function_declaration.decl);
            walk_list!(visitor, visit_lifetime_def, &function_declaration.lifetimes);
        }
        TyPath(ref qpath) => {
            visitor.visit_qpath(qpath, typ.id, typ.span);
        }
        TyArray(ref ty, length) => {
            visitor.visit_ty(ty);
            visitor.visit_nested_body(length)
        }
        TyTraitObject(ref bounds) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
        }
        TyImplTrait(ref bounds) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
        }
        TyTypeof(expression) => {
            visitor.visit_nested_body(expression)
        }
        TyInfer => {}
    }
}

pub fn walk_qpath<'v, V: Visitor<'v>>(visitor: &mut V, qpath: &'v QPath, id: NodeId, span: Span) {
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
    visitor.visit_def_mention(path.def);
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_path_segment<'v, V: Visitor<'v>>(visitor: &mut V,
                                             path_span: Span,
                                             segment: &'v PathSegment) {
    visitor.visit_name(path_span, segment.name);
    visitor.visit_path_parameters(path_span, &segment.parameters);
}

pub fn walk_path_parameters<'v, V: Visitor<'v>>(visitor: &mut V,
                                                _path_span: Span,
                                                path_parameters: &'v PathParameters) {
    match *path_parameters {
        AngleBracketedParameters(ref data) => {
            walk_list!(visitor, visit_ty, &data.types);
            walk_list!(visitor, visit_lifetime, &data.lifetimes);
            walk_list!(visitor, visit_assoc_type_binding, &data.bindings);
        }
        ParenthesizedParameters(ref data) => {
            walk_list!(visitor, visit_ty, &data.inputs);
            walk_list!(visitor, visit_ty, &data.output);
        }
    }
}

pub fn walk_assoc_type_binding<'v, V: Visitor<'v>>(visitor: &mut V,
                                                   type_binding: &'v TypeBinding) {
    visitor.visit_id(type_binding.id);
    visitor.visit_name(type_binding.span, type_binding.name);
    visitor.visit_ty(&type_binding.ty);
}

pub fn walk_pat<'v, V: Visitor<'v>>(visitor: &mut V, pattern: &'v Pat) {
    visitor.visit_id(pattern.id);
    match pattern.node {
        PatKind::TupleStruct(ref qpath, ref children, _) => {
            visitor.visit_qpath(qpath, pattern.id, pattern.span);
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref qpath) => {
            visitor.visit_qpath(qpath, pattern.id, pattern.span);
        }
        PatKind::Struct(ref qpath, ref fields, _) => {
            visitor.visit_qpath(qpath, pattern.id, pattern.span);
            for field in fields {
                visitor.visit_name(field.span, field.node.name);
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
        PatKind::Binding(_, def_id, ref pth1, ref optional_subpattern) => {
            visitor.visit_def_mention(Def::Local(def_id));
            visitor.visit_name(pth1.span, pth1.node);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => visitor.visit_expr(expression),
        PatKind::Range(ref lower_bound, ref upper_bound) => {
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
    visitor.visit_id(foreign_item.id);
    visitor.visit_vis(&foreign_item.vis);
    visitor.visit_name(foreign_item.span, foreign_item.name);

    match foreign_item.node {
        ForeignItemFn(ref function_declaration, ref names, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_fn_decl(function_declaration);
            for name in names {
                visitor.visit_name(name.span, name.node);
            }
        }
        ForeignItemStatic(ref typ, _) => visitor.visit_ty(typ),
    }

    walk_list!(visitor, visit_attribute, &foreign_item.attrs);
}

pub fn walk_ty_param_bound<'v, V: Visitor<'v>>(visitor: &mut V, bound: &'v TyParamBound) {
    match *bound {
        TraitTyParamBound(ref typ, ref modifier) => {
            visitor.visit_poly_trait_ref(typ, modifier);
        }
        RegionTyParamBound(ref lifetime) => {
            visitor.visit_lifetime(lifetime);
        }
    }
}

pub fn walk_generics<'v, V: Visitor<'v>>(visitor: &mut V, generics: &'v Generics) {
    for param in &generics.ty_params {
        visitor.visit_id(param.id);
        visitor.visit_name(param.span, param.name);
        walk_list!(visitor, visit_ty_param_bound, &param.bounds);
        walk_list!(visitor, visit_ty, &param.default);
    }
    walk_list!(visitor, visit_lifetime_def, &generics.lifetimes);
    visitor.visit_id(generics.where_clause.id);
    walk_list!(visitor, visit_where_predicate, &generics.where_clause.predicates);
}

pub fn walk_where_predicate<'v, V: Visitor<'v>>(
    visitor: &mut V,
    predicate: &'v WherePredicate)
{
    match predicate {
        &WherePredicate::BoundPredicate(WhereBoundPredicate{ref bounded_ty,
                                                            ref bounds,
                                                            ref bound_lifetimes,
                                                            ..}) => {
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_lifetime_def, bound_lifetimes);
        }
        &WherePredicate::RegionPredicate(WhereRegionPredicate{ref lifetime,
                                                              ref bounds,
                                                              ..}) => {
            visitor.visit_lifetime(lifetime);
            walk_list!(visitor, visit_lifetime, bounds);
        }
        &WherePredicate::EqPredicate(WhereEqPredicate{id,
                                                      ref lhs_ty,
                                                      ref rhs_ty,
                                                      ..}) => {
            visitor.visit_id(id);
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
        FnKind::Method(_, sig, ..) => {
            visitor.visit_generics(&sig.generics);
        }
        FnKind::Closure(_) => {}
    }
}

pub fn walk_fn<'v, V: Visitor<'v>>(visitor: &mut V,
                                   function_kind: FnKind<'v>,
                                   function_declaration: &'v FnDecl,
                                   body_id: BodyId,
                                   _span: Span,
                                   id: NodeId) {
    visitor.visit_id(id);
    visitor.visit_fn_decl(function_declaration);
    walk_fn_kind(visitor, function_kind);
    visitor.visit_nested_body(body_id)
}

pub fn walk_trait_item<'v, V: Visitor<'v>>(visitor: &mut V, trait_item: &'v TraitItem) {
    visitor.visit_name(trait_item.span, trait_item.name);
    walk_list!(visitor, visit_attribute, &trait_item.attrs);
    match trait_item.node {
        TraitItemKind::Const(ref ty, default) => {
            visitor.visit_id(trait_item.id);
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_nested_body, default);
        }
        TraitItemKind::Method(ref sig, TraitMethod::Required(ref names)) => {
            visitor.visit_id(trait_item.id);
            visitor.visit_generics(&sig.generics);
            visitor.visit_fn_decl(&sig.decl);
            for name in names {
                visitor.visit_name(name.span, name.node);
            }
        }
        TraitItemKind::Method(ref sig, TraitMethod::Provided(body_id)) => {
            visitor.visit_fn(FnKind::Method(trait_item.name,
                                            sig,
                                            None,
                                            &trait_item.attrs),
                             &sig.decl,
                             body_id,
                             trait_item.span,
                             trait_item.id);
        }
        TraitItemKind::Type(ref bounds, ref default) => {
            visitor.visit_id(trait_item.id);
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
    }
}

pub fn walk_trait_item_ref<'v, V: Visitor<'v>>(visitor: &mut V, trait_item_ref: &'v TraitItemRef) {
    // NB: Deliberately force a compilation error if/when new fields are added.
    let TraitItemRef { id, name, ref kind, span, ref defaultness } = *trait_item_ref;
    visitor.visit_nested_trait_item(id);
    visitor.visit_name(span, name);
    visitor.visit_associated_item_kind(kind);
    visitor.visit_defaultness(defaultness);
}

pub fn walk_impl_item<'v, V: Visitor<'v>>(visitor: &mut V, impl_item: &'v ImplItem) {
    // NB: Deliberately force a compilation error if/when new fields are added.
    let ImplItem { id: _, name, ref vis, ref defaultness, ref attrs, ref node, span } = *impl_item;

    visitor.visit_name(span, name);
    visitor.visit_vis(vis);
    visitor.visit_defaultness(defaultness);
    walk_list!(visitor, visit_attribute, attrs);
    match *node {
        ImplItemKind::Const(ref ty, body) => {
            visitor.visit_id(impl_item.id);
            visitor.visit_ty(ty);
            visitor.visit_nested_body(body);
        }
        ImplItemKind::Method(ref sig, body_id) => {
            visitor.visit_fn(FnKind::Method(impl_item.name,
                                            sig,
                                            Some(&impl_item.vis),
                                            &impl_item.attrs),
                             &sig.decl,
                             body_id,
                             impl_item.span,
                             impl_item.id);
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_id(impl_item.id);
            visitor.visit_ty(ty);
        }
    }
}

pub fn walk_impl_item_ref<'v, V: Visitor<'v>>(visitor: &mut V, impl_item_ref: &'v ImplItemRef) {
    // NB: Deliberately force a compilation error if/when new fields are added.
    let ImplItemRef { id, name, ref kind, span, ref vis, ref defaultness } = *impl_item_ref;
    visitor.visit_nested_impl_item(id);
    visitor.visit_name(span, name);
    visitor.visit_associated_item_kind(kind);
    visitor.visit_vis(vis);
    visitor.visit_defaultness(defaultness);
}


pub fn walk_struct_def<'v, V: Visitor<'v>>(visitor: &mut V, struct_definition: &'v VariantData) {
    visitor.visit_id(struct_definition.id());
    walk_list!(visitor, visit_struct_field, struct_definition.fields());
}

pub fn walk_struct_field<'v, V: Visitor<'v>>(visitor: &mut V, struct_field: &'v StructField) {
    visitor.visit_id(struct_field.id);
    visitor.visit_vis(&struct_field.vis);
    visitor.visit_name(struct_field.span, struct_field.name);
    visitor.visit_ty(&struct_field.ty);
    walk_list!(visitor, visit_attribute, &struct_field.attrs);
}

pub fn walk_block<'v, V: Visitor<'v>>(visitor: &mut V, block: &'v Block) {
    visitor.visit_id(block.id);
    walk_list!(visitor, visit_stmt, &block.stmts);
    walk_list!(visitor, visit_expr, &block.expr);
}

pub fn walk_stmt<'v, V: Visitor<'v>>(visitor: &mut V, statement: &'v Stmt) {
    match statement.node {
        StmtDecl(ref declaration, id) => {
            visitor.visit_id(id);
            visitor.visit_decl(declaration)
        }
        StmtExpr(ref expression, id) |
        StmtSemi(ref expression, id) => {
            visitor.visit_id(id);
            visitor.visit_expr(expression)
        }
    }
}

pub fn walk_decl<'v, V: Visitor<'v>>(visitor: &mut V, declaration: &'v Decl) {
    match declaration.node {
        DeclLocal(ref local) => visitor.visit_local(local),
        DeclItem(item) => visitor.visit_nested_item(item),
    }
}

pub fn walk_expr<'v, V: Visitor<'v>>(visitor: &mut V, expression: &'v Expr) {
    visitor.visit_id(expression.id);
    match expression.node {
        ExprBox(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprArray(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprRepeat(ref element, count) => {
            visitor.visit_expr(element);
            visitor.visit_nested_body(count)
        }
        ExprStruct(ref qpath, ref fields, ref optional_base) => {
            visitor.visit_qpath(qpath, expression.id, expression.span);
            for field in fields {
                visitor.visit_name(field.name.span, field.name.node);
                visitor.visit_expr(&field.expr)
            }
            walk_list!(visitor, visit_expr, optional_base);
        }
        ExprTup(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprCall(ref callee_expression, ref arguments) => {
            walk_list!(visitor, visit_expr, arguments);
            visitor.visit_expr(callee_expression)
        }
        ExprMethodCall(ref name, ref types, ref arguments) => {
            visitor.visit_name(name.span, name.node);
            walk_list!(visitor, visit_expr, arguments);
            walk_list!(visitor, visit_ty, types);
        }
        ExprBinary(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprAddrOf(_, ref subexpression) | ExprUnary(_, ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprLit(_) => {}
        ExprCast(ref subexpression, ref typ) | ExprType(ref subexpression, ref typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprIf(ref head_expression, ref if_block, ref optional_else) => {
            visitor.visit_expr(head_expression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprWhile(ref subexpression, ref block, ref opt_sp_name) => {
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_sp_name(visitor, opt_sp_name);
        }
        ExprLoop(ref block, ref opt_sp_name, _) => {
            visitor.visit_block(block);
            walk_opt_sp_name(visitor, opt_sp_name);
        }
        ExprMatch(ref subexpression, ref arms, _) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprClosure(_, ref function_declaration, body, _fn_decl_span) => {
            visitor.visit_fn(FnKind::Closure(&expression.attrs),
                             function_declaration,
                             body,
                             expression.span,
                             expression.id)
        }
        ExprBlock(ref block) => visitor.visit_block(block),
        ExprAssign(ref left_hand_expression, ref right_hand_expression) => {
            visitor.visit_expr(right_hand_expression);
            visitor.visit_expr(left_hand_expression)
        }
        ExprAssignOp(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(right_expression);
            visitor.visit_expr(left_expression)
        }
        ExprField(ref subexpression, ref name) => {
            visitor.visit_expr(subexpression);
            visitor.visit_name(name.span, name.node);
        }
        ExprTupField(ref subexpression, _) => {
            visitor.visit_expr(subexpression);
        }
        ExprIndex(ref main_expression, ref index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprPath(ref qpath) => {
            visitor.visit_qpath(qpath, expression.id, expression.span);
        }
        ExprBreak(None, ref opt_expr) => {
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprBreak(Some(label), ref opt_expr) => {
            visitor.visit_def_mention(Def::Label(label.loop_id));
            visitor.visit_name(label.span, label.name);
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprAgain(None) => {}
        ExprAgain(Some(label)) => {
            visitor.visit_def_mention(Def::Label(label.loop_id));
            visitor.visit_name(label.span, label.name);
        }
        ExprRet(ref optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprInlineAsm(_, ref outputs, ref inputs) => {
            for output in outputs {
                visitor.visit_expr(output)
            }
            for input in inputs {
                visitor.visit_expr(input)
            }
        }
    }
}

pub fn walk_arm<'v, V: Visitor<'v>>(visitor: &mut V, arm: &'v Arm) {
    walk_list!(visitor, visit_pat, &arm.pats);
    walk_list!(visitor, visit_expr, &arm.guard);
    visitor.visit_expr(&arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
}

pub fn walk_vis<'v, V: Visitor<'v>>(visitor: &mut V, vis: &'v Visibility) {
    if let Visibility::Restricted { ref path, id } = *vis {
        visitor.visit_id(id);
        visitor.visit_path(path, id)
    }
}

pub fn walk_associated_item_kind<'v, V: Visitor<'v>>(_: &mut V, _: &'v AssociatedItemKind) {
    // No visitable content here: this fn exists so you can call it if
    // the right thing to do, should content be added in the future,
    // would be to walk it.
}

pub fn walk_defaultness<'v, V: Visitor<'v>>(_: &mut V, _: &'v Defaultness) {
    // No visitable content here: this fn exists so you can call it if
    // the right thing to do, should content be added in the future,
    // would be to walk it.
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, PartialEq, Eq)]
pub struct IdRange {
    pub min: NodeId,
    pub max: NodeId,
}

impl IdRange {
    pub fn max() -> IdRange {
        IdRange {
            min: NodeId::from_u32(u32::MAX),
            max: NodeId::from_u32(u32::MIN),
        }
    }

    pub fn empty(&self) -> bool {
        self.min >= self.max
    }

    pub fn contains(&self, id: NodeId) -> bool {
        id >= self.min && id < self.max
    }

    pub fn add(&mut self, id: NodeId) {
        self.min = cmp::min(self.min, id);
        self.max = cmp::max(self.max, NodeId::from_u32(id.as_u32() + 1));
    }

}


pub struct IdRangeComputingVisitor<'a, 'ast: 'a> {
    result: IdRange,
    map: &'a map::Map<'ast>,
}

impl<'a, 'ast> IdRangeComputingVisitor<'a, 'ast> {
    pub fn new(map: &'a map::Map<'ast>) -> IdRangeComputingVisitor<'a, 'ast> {
        IdRangeComputingVisitor { result: IdRange::max(), map: map }
    }

    pub fn result(&self) -> IdRange {
        self.result
    }
}

impl<'a, 'ast> Visitor<'ast> for IdRangeComputingVisitor<'a, 'ast> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'ast> {
        NestedVisitorMap::OnlyBodies(&self.map)
    }

    fn visit_id(&mut self, id: NodeId) {
        self.result.add(id);
    }
}
