// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! HIR walker. Each overridden visit method has full control over what
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
use syntax::attr::ThinAttributesExt;
use syntax::codemap::Span;
use hir::*;

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
            FnKind::ItemFn(_, _, _, _, _, _, attrs) => attrs,
            FnKind::Method(_, _, _, attrs) => attrs,
            FnKind::Closure(attrs) => attrs,
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

    /// Invoked when a nested item is encountered. By default, does
    /// nothing. If you want a deep walk, you need to override to
    /// fetch the item contents. But most of the time, it is easier
    /// (and better) to invoke `Crate::visit_all_items`, which visits
    /// all items in the crate in some order (but doesn't respect
    /// nesting).
    #[allow(unused_variables)]
    fn visit_nested_item(&mut self, id: ItemId) {
    }

    /// Visit the top-level item and (optionally) nested items. See
    /// `visit_nested_item` for details.
    fn visit_item(&mut self, i: &'v Item) {
        walk_item(self, i)
    }

    ///////////////////////////////////////////////////////////////////////////

    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, span: Span, ident: Ident) {
        walk_ident(self, span, ident);
    }
    fn visit_mod(&mut self, m: &'v Mod, _s: Span, _n: NodeId) {
        walk_mod(self, m)
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
    fn visit_expr_post(&mut self, _ex: &'v Expr) {
    }
    fn visit_ty(&mut self, t: &'v Ty) {
        walk_ty(self, t)
    }
    fn visit_generics(&mut self, g: &'v Generics) {
        walk_generics(self, g)
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: &'v Block, s: Span, _: NodeId) {
        walk_fn(self, fk, fd, b, s)
    }
    fn visit_trait_item(&mut self, ti: &'v TraitItem) {
        walk_trait_item(self, ti)
    }
    fn visit_impl_item(&mut self, ii: &'v ImplItem) {
        walk_impl_item(self, ii)
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
                          _: NodeId,
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
    fn visit_explicit_self(&mut self, es: &'v ExplicitSelf) {
        walk_explicit_self(self, es)
    }
    fn visit_path(&mut self, path: &'v Path, _id: NodeId) {
        walk_path(self, path)
    }
    fn visit_path_list_item(&mut self, prefix: &'v Path, item: &'v PathListItem) {
        walk_path_list_item(self, prefix, item)
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
}

pub fn walk_opt_name<'v, V: Visitor<'v>>(visitor: &mut V, span: Span, opt_name: Option<Name>) {
    for name in opt_name {
        visitor.visit_name(span, name);
    }
}

pub fn walk_opt_ident<'v, V: Visitor<'v>>(visitor: &mut V, span: Span, opt_ident: Option<Ident>) {
    for ident in opt_ident {
        visitor.visit_ident(span, ident);
    }
}

pub fn walk_ident<'v, V: Visitor<'v>>(visitor: &mut V, span: Span, ident: Ident) {
    visitor.visit_name(span, ident.name);
}

/// Walks the contents of a crate. See also `Crate::visit_all_items`.
pub fn walk_crate<'v, V: Visitor<'v>>(visitor: &mut V, krate: &'v Crate) {
    visitor.visit_mod(&krate.module, krate.span, CRATE_NODE_ID);
    walk_list!(visitor, visit_attribute, &krate.attrs);
    walk_list!(visitor, visit_macro_def, &krate.exported_macros);
}

pub fn walk_macro_def<'v, V: Visitor<'v>>(visitor: &mut V, macro_def: &'v MacroDef) {
    visitor.visit_name(macro_def.span, macro_def.name);
    walk_opt_name(visitor, macro_def.span, macro_def.imported_from);
    walk_list!(visitor, visit_attribute, &macro_def.attrs);
}

pub fn walk_mod<'v, V: Visitor<'v>>(visitor: &mut V, module: &'v Mod) {
    for &item_id in &module.item_ids {
        visitor.visit_nested_item(item_id);
    }
}

pub fn walk_local<'v, V: Visitor<'v>>(visitor: &mut V, local: &'v Local) {
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
    walk_list!(visitor, visit_expr, &local.init);
}

pub fn walk_lifetime<'v, V: Visitor<'v>>(visitor: &mut V, lifetime: &'v Lifetime) {
    visitor.visit_name(lifetime.span, lifetime.name);
}

pub fn walk_lifetime_def<'v, V: Visitor<'v>>(visitor: &mut V, lifetime_def: &'v LifetimeDef) {
    visitor.visit_lifetime(&lifetime_def.lifetime);
    walk_list!(visitor, visit_lifetime, &lifetime_def.bounds);
}

pub fn walk_explicit_self<'v, V: Visitor<'v>>(visitor: &mut V, explicit_self: &'v ExplicitSelf) {
    match explicit_self.node {
        SelfStatic => {}
        SelfValue(name) => {
            visitor.visit_name(explicit_self.span, name)
        }
        SelfRegion(ref opt_lifetime, _, name) => {
            visitor.visit_name(explicit_self.span, name);
            walk_list!(visitor, visit_lifetime, opt_lifetime);
        }
        SelfExplicit(ref typ, name) => {
            visitor.visit_name(explicit_self.span, name);
            visitor.visit_ty(typ)
        }
    }
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
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<'v, V: Visitor<'v>>(visitor: &mut V, item: &'v Item) {
    visitor.visit_vis(&item.vis);
    visitor.visit_name(item.span, item.name);
    match item.node {
        ItemExternCrate(opt_name) => {
            walk_opt_name(visitor, item.span, opt_name)
        }
        ItemUse(ref vp) => {
            match vp.node {
                ViewPathSimple(name, ref path) => {
                    visitor.visit_name(vp.span, name);
                    visitor.visit_path(path, item.id);
                }
                ViewPathGlob(ref path) => {
                    visitor.visit_path(path, item.id);
                }
                ViewPathList(ref prefix, ref list) => {
                    if !list.is_empty() {
                        for item in list {
                            visitor.visit_path_list_item(prefix, item)
                        }
                    } else {
                        visitor.visit_path(prefix, item.id);
                    }
                }
            }
        }
        ItemStatic(ref typ, _, ref expr) |
        ItemConst(ref typ, ref expr) => {
            visitor.visit_ty(typ);
            visitor.visit_expr(expr);
        }
        ItemFn(ref declaration, unsafety, constness, abi, ref generics, ref body) => {
            visitor.visit_fn(FnKind::ItemFn(item.name,
                                            generics,
                                            unsafety,
                                            constness,
                                            abi,
                                            &item.vis,
                                            &item.attrs),
                             declaration,
                             body,
                             item.span,
                             item.id)
        }
        ItemMod(ref module) => {
            visitor.visit_mod(module, item.span, item.id)
        }
        ItemForeignMod(ref foreign_module) => {
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemTy(ref typ, ref type_parameters) => {
            visitor.visit_ty(typ);
            visitor.visit_generics(type_parameters)
        }
        ItemEnum(ref enum_definition, ref type_parameters) => {
            visitor.visit_generics(type_parameters);
            visitor.visit_enum_def(enum_definition, type_parameters, item.id, item.span)
        }
        ItemDefaultImpl(_, ref trait_ref) => {
            visitor.visit_trait_ref(trait_ref)
        }
        ItemImpl(_, _, ref type_parameters, ref opt_trait_reference, ref typ, ref impl_items) => {
            visitor.visit_generics(type_parameters);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_impl_item, impl_items);
        }
        ItemStruct(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition, item.name, generics, item.id, item.span);
        }
        ItemTrait(_, ref generics, ref bounds, ref methods) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_trait_item, methods);
        }
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'v, V: Visitor<'v>>(visitor: &mut V,
                                         enum_definition: &'v EnumDef,
                                         generics: &'v Generics,
                                         item_id: NodeId) {
    walk_list!(visitor,
               visit_variant,
               &enum_definition.variants,
               generics,
               item_id);
}

pub fn walk_variant<'v, V: Visitor<'v>>(visitor: &mut V,
                                        variant: &'v Variant,
                                        generics: &'v Generics,
                                        item_id: NodeId) {
    visitor.visit_name(variant.span, variant.node.name);
    visitor.visit_variant_data(&variant.node.data,
                               variant.node.name,
                               generics,
                               item_id,
                               variant.span);
    walk_list!(visitor, visit_expr, &variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.node.attrs);
}

pub fn walk_ty<'v, V: Visitor<'v>>(visitor: &mut V, typ: &'v Ty) {
    match typ.node {
        TyVec(ref ty) => {
            visitor.visit_ty(ty)
        }
        TyPtr(ref mutable_type) => {
            visitor.visit_ty(&mutable_type.ty)
        }
        TyRptr(ref opt_lifetime, ref mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyTup(ref tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyBareFn(ref function_declaration) => {
            walk_fn_decl(visitor, &function_declaration.decl);
            walk_list!(visitor, visit_lifetime_def, &function_declaration.lifetimes);
        }
        TyPath(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, typ.id);
        }
        TyObjectSum(ref ty, ref bounds) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_ty_param_bound, bounds);
        }
        TyFixedLengthVec(ref ty, ref expression) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expression)
        }
        TyPolyTraitRef(ref bounds) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
        }
        TyTypeof(ref expression) => {
            visitor.visit_expr(expression)
        }
        TyInfer => {}
    }
}

pub fn walk_path<'v, V: Visitor<'v>>(visitor: &mut V, path: &'v Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_path_list_item<'v, V: Visitor<'v>>(visitor: &mut V,
                                               prefix: &'v Path,
                                               item: &'v PathListItem) {
    for segment in &prefix.segments {
        visitor.visit_path_segment(prefix.span, segment);
    }

    walk_opt_name(visitor, item.span, item.node.name());
    walk_opt_name(visitor, item.span, item.node.rename());
}

pub fn walk_path_segment<'v, V: Visitor<'v>>(visitor: &mut V,
                                             path_span: Span,
                                             segment: &'v PathSegment) {
    visitor.visit_ident(path_span, segment.identifier);
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
    visitor.visit_name(type_binding.span, type_binding.name);
    visitor.visit_ty(&type_binding.ty);
}

pub fn walk_pat<'v, V: Visitor<'v>>(visitor: &mut V, pattern: &'v Pat) {
    match pattern.node {
        PatKind::TupleStruct(ref path, ref opt_children) => {
            visitor.visit_path(path, pattern.id);
            if let Some(ref children) = *opt_children {
                walk_list!(visitor, visit_pat, children);
            }
        }
        PatKind::Path(ref path) => {
            visitor.visit_path(path, pattern.id);
        }
        PatKind::QPath(ref qself, ref path) => {
            visitor.visit_ty(&qself.ty);
            visitor.visit_path(path, pattern.id)
        }
        PatKind::Struct(ref path, ref fields, _) => {
            visitor.visit_path(path, pattern.id);
            for field in fields {
                visitor.visit_name(field.span, field.node.name);
                visitor.visit_pat(&field.node.pat)
            }
        }
        PatKind::Tup(ref tuple_elements) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatKind::Box(ref subpattern) |
        PatKind::Ref(ref subpattern, _) => {
            visitor.visit_pat(subpattern)
        }
        PatKind::Ident(_, ref pth1, ref optional_subpattern) => {
            visitor.visit_ident(pth1.span, pth1.node);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => visitor.visit_expr(expression),
        PatKind::Range(ref lower_bound, ref upper_bound) => {
            visitor.visit_expr(lower_bound);
            visitor.visit_expr(upper_bound)
        }
        PatKind::Wild => (),
        PatKind::Vec(ref prepatterns, ref slice_pattern, ref postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
    }
}

pub fn walk_foreign_item<'v, V: Visitor<'v>>(visitor: &mut V, foreign_item: &'v ForeignItem) {
    visitor.visit_vis(&foreign_item.vis);
    visitor.visit_name(foreign_item.span, foreign_item.name);

    match foreign_item.node {
        ForeignItemFn(ref function_declaration, ref generics) => {
            walk_fn_decl(visitor, function_declaration);
            visitor.visit_generics(generics)
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
        visitor.visit_name(param.span, param.name);
        walk_list!(visitor, visit_ty_param_bound, &param.bounds);
        walk_list!(visitor, visit_ty, &param.default);
    }
    walk_list!(visitor, visit_lifetime_def, &generics.lifetimes);
    for predicate in &generics.where_clause.predicates {
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
                                                                    ref path,
                                                                    ref ty,
                                                                    ..}) => {
                visitor.visit_path(path, id);
                visitor.visit_ty(ty);
            }
        }
    }
}

pub fn walk_fn_ret_ty<'v, V: Visitor<'v>>(visitor: &mut V, ret_ty: &'v FunctionRetTy) {
    if let Return(ref output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<'v, V: Visitor<'v>>(visitor: &mut V, function_declaration: &'v FnDecl) {
    for argument in &function_declaration.inputs {
        visitor.visit_pat(&argument.pat);
        visitor.visit_ty(&argument.ty)
    }
    walk_fn_ret_ty(visitor, &function_declaration.output)
}

pub fn walk_fn_decl_nopat<'v, V: Visitor<'v>>(visitor: &mut V, function_declaration: &'v FnDecl) {
    for argument in &function_declaration.inputs {
        visitor.visit_ty(&argument.ty)
    }
    walk_fn_ret_ty(visitor, &function_declaration.output)
}

pub fn walk_fn_kind<'v, V: Visitor<'v>>(visitor: &mut V, function_kind: FnKind<'v>) {
    match function_kind {
        FnKind::ItemFn(_, generics, _, _, _, _, _) => {
            visitor.visit_generics(generics);
        }
        FnKind::Method(_, sig, _, _) => {
            visitor.visit_generics(&sig.generics);
            visitor.visit_explicit_self(&sig.explicit_self);
        }
        FnKind::Closure(_) => {}
    }
}

pub fn walk_fn<'v, V: Visitor<'v>>(visitor: &mut V,
                                   function_kind: FnKind<'v>,
                                   function_declaration: &'v FnDecl,
                                   function_body: &'v Block,
                                   _span: Span) {
    walk_fn_decl(visitor, function_declaration);
    walk_fn_kind(visitor, function_kind);
    visitor.visit_block(function_body)
}

pub fn walk_trait_item<'v, V: Visitor<'v>>(visitor: &mut V, trait_item: &'v TraitItem) {
    visitor.visit_name(trait_item.span, trait_item.name);
    walk_list!(visitor, visit_attribute, &trait_item.attrs);
    match trait_item.node {
        ConstTraitItem(ref ty, ref default) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, default);
        }
        MethodTraitItem(ref sig, None) => {
            visitor.visit_explicit_self(&sig.explicit_self);
            visitor.visit_generics(&sig.generics);
            walk_fn_decl(visitor, &sig.decl);
        }
        MethodTraitItem(ref sig, Some(ref body)) => {
            visitor.visit_fn(FnKind::Method(trait_item.name,
                                            sig,
                                            None,
                                            &trait_item.attrs),
                             &sig.decl,
                             body,
                             trait_item.span,
                             trait_item.id);
        }
        TypeTraitItem(ref bounds, ref default) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
    }
}

pub fn walk_impl_item<'v, V: Visitor<'v>>(visitor: &mut V, impl_item: &'v ImplItem) {
    visitor.visit_vis(&impl_item.vis);
    visitor.visit_name(impl_item.span, impl_item.name);
    walk_list!(visitor, visit_attribute, &impl_item.attrs);
    match impl_item.node {
        ImplItemKind::Const(ref ty, ref expr) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expr);
        }
        ImplItemKind::Method(ref sig, ref body) => {
            visitor.visit_fn(FnKind::Method(impl_item.name,
                                            sig,
                                            Some(&impl_item.vis),
                                            &impl_item.attrs),
                             &sig.decl,
                             body,
                             impl_item.span,
                             impl_item.id);
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_ty(ty);
        }
    }
}

pub fn walk_struct_def<'v, V: Visitor<'v>>(visitor: &mut V, struct_definition: &'v VariantData) {
    walk_list!(visitor, visit_struct_field, struct_definition.fields());
}

pub fn walk_struct_field<'v, V: Visitor<'v>>(visitor: &mut V, struct_field: &'v StructField) {
    visitor.visit_vis(&struct_field.vis);
    visitor.visit_name(struct_field.span, struct_field.name);
    visitor.visit_ty(&struct_field.ty);
    walk_list!(visitor, visit_attribute, &struct_field.attrs);
}

pub fn walk_block<'v, V: Visitor<'v>>(visitor: &mut V, block: &'v Block) {
    walk_list!(visitor, visit_stmt, &block.stmts);
    walk_list!(visitor, visit_expr, &block.expr);
}

pub fn walk_stmt<'v, V: Visitor<'v>>(visitor: &mut V, statement: &'v Stmt) {
    match statement.node {
        StmtDecl(ref declaration, _) => visitor.visit_decl(declaration),
        StmtExpr(ref expression, _) | StmtSemi(ref expression, _) => {
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
    match expression.node {
        ExprBox(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprVec(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprRepeat(ref element, ref count) => {
            visitor.visit_expr(element);
            visitor.visit_expr(count)
        }
        ExprStruct(ref path, ref fields, ref optional_base) => {
            visitor.visit_path(path, expression.id);
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
        ExprWhile(ref subexpression, ref block, opt_ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_ident(visitor, expression.span, opt_ident)
        }
        ExprLoop(ref block, opt_ident) => {
            visitor.visit_block(block);
            walk_opt_ident(visitor, expression.span, opt_ident)
        }
        ExprMatch(ref subexpression, ref arms, _) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprClosure(_, ref function_declaration, ref body, _fn_decl_span) => {
            visitor.visit_fn(FnKind::Closure(expression.attrs.as_attr_slice()),
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
        ExprPath(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, expression.id)
        }
        ExprBreak(ref opt_sp_ident) | ExprAgain(ref opt_sp_ident) => {
            for sp_ident in opt_sp_ident {
                visitor.visit_ident(sp_ident.span, sp_ident.node);
            }
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

    visitor.visit_expr_post(expression)
}

pub fn walk_arm<'v, V: Visitor<'v>>(visitor: &mut V, arm: &'v Arm) {
    walk_list!(visitor, visit_pat, &arm.pats);
    walk_list!(visitor, visit_expr, &arm.guard);
    visitor.visit_expr(&arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
}

pub fn walk_vis<'v, V: Visitor<'v>>(visitor: &mut V, vis: &'v Visibility) {
    if let Visibility::Restricted { ref path, id } = *vis {
        visitor.visit_path(path, id)
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct IdRange {
    pub min: NodeId,
    pub max: NodeId,
}

impl IdRange {
    pub fn max() -> IdRange {
        IdRange {
            min: u32::MAX,
            max: u32::MIN,
        }
    }

    pub fn empty(&self) -> bool {
        self.min >= self.max
    }

    pub fn add(&mut self, id: NodeId) {
        self.min = cmp::min(self.min, id);
        self.max = cmp::max(self.max, id + 1);
    }
}

pub trait IdVisitingOperation {
    fn visit_id(&mut self, node_id: NodeId);
}

pub struct IdRangeComputingVisitor {
    pub result: IdRange,
}

impl IdRangeComputingVisitor {
    pub fn new() -> IdRangeComputingVisitor {
        IdRangeComputingVisitor { result: IdRange::max() }
    }

    pub fn result(&self) -> IdRange {
        self.result
    }
}

impl IdVisitingOperation for IdRangeComputingVisitor {
    fn visit_id(&mut self, id: NodeId) {
        self.result.add(id);
    }
}

pub struct IdVisitor<'a, O: 'a> {
    operation: &'a mut O,

    // In general, the id visitor visits the contents of an item, but
    // not including nested trait/impl items, nor other nested items.
    // The base visitor itself always skips nested items, but not
    // trait/impl items. This means in particular that if you start by
    // visiting a trait or an impl, you should not visit the
    // trait/impl items respectively.  This is handled by setting
    // `skip_members` to true when `visit_item` is on the stack. This
    // way, if the user begins by calling `visit_trait_item`, we will
    // visit the trait item, but if they begin with `visit_item`, we
    // won't visit the (nested) trait items.
    skip_members: bool,
}

impl<'a, O: IdVisitingOperation> IdVisitor<'a, O> {
    pub fn new(operation: &'a mut O) -> IdVisitor<'a, O> {
        IdVisitor { operation: operation, skip_members: false }
    }

    fn visit_generics_helper(&mut self, generics: &Generics) {
        for type_parameter in generics.ty_params.iter() {
            self.operation.visit_id(type_parameter.id)
        }
        for lifetime in &generics.lifetimes {
            self.operation.visit_id(lifetime.lifetime.id)
        }
    }
}

impl<'a, 'v, O: IdVisitingOperation> Visitor<'v> for IdVisitor<'a, O> {
    fn visit_mod(&mut self, module: &Mod, _: Span, node_id: NodeId) {
        self.operation.visit_id(node_id);
        walk_mod(self, module)
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.operation.visit_id(foreign_item.id);
        walk_foreign_item(self, foreign_item)
    }

    fn visit_item(&mut self, item: &Item) {
        assert!(!self.skip_members);
        self.skip_members = true;

        self.operation.visit_id(item.id);
        match item.node {
            ItemUse(ref view_path) => {
                match view_path.node {
                    ViewPathSimple(_, _) |
                    ViewPathGlob(_) => {}
                    ViewPathList(_, ref paths) => {
                        for path in paths {
                            self.operation.visit_id(path.node.id())
                        }
                    }
                }
            }
            _ => {}
        }
        walk_item(self, item);

        self.skip_members = false;
    }

    fn visit_local(&mut self, local: &Local) {
        self.operation.visit_id(local.id);
        walk_local(self, local)
    }

    fn visit_block(&mut self, block: &Block) {
        self.operation.visit_id(block.id);
        walk_block(self, block)
    }

    fn visit_stmt(&mut self, statement: &Stmt) {
        self.operation.visit_id(statement.node.id());
        walk_stmt(self, statement)
    }

    fn visit_pat(&mut self, pattern: &Pat) {
        self.operation.visit_id(pattern.id);
        walk_pat(self, pattern)
    }

    fn visit_expr(&mut self, expression: &Expr) {
        self.operation.visit_id(expression.id);
        walk_expr(self, expression)
    }

    fn visit_ty(&mut self, typ: &Ty) {
        self.operation.visit_id(typ.id);
        walk_ty(self, typ)
    }

    fn visit_generics(&mut self, generics: &Generics) {
        self.visit_generics_helper(generics);
        walk_generics(self, generics)
    }

    fn visit_fn(&mut self,
                function_kind: FnKind<'v>,
                function_declaration: &'v FnDecl,
                block: &'v Block,
                span: Span,
                node_id: NodeId) {
        self.operation.visit_id(node_id);

        match function_kind {
            FnKind::ItemFn(_, generics, _, _, _, _, _) => {
                self.visit_generics_helper(generics)
            }
            FnKind::Method(_, sig, _, _) => {
                self.visit_generics_helper(&sig.generics)
            }
            FnKind::Closure(_) => {}
        }

        for argument in &function_declaration.inputs {
            self.operation.visit_id(argument.id)
        }

        walk_fn(self, function_kind, function_declaration, block, span);
    }

    fn visit_struct_field(&mut self, struct_field: &StructField) {
        self.operation.visit_id(struct_field.id);
        walk_struct_field(self, struct_field)
    }

    fn visit_variant_data(&mut self,
                          struct_def: &VariantData,
                          _: Name,
                          _: &Generics,
                          _: NodeId,
                          _: Span) {
        self.operation.visit_id(struct_def.id());
        walk_struct_def(self, struct_def);
    }

    fn visit_trait_item(&mut self, ti: &TraitItem) {
        if !self.skip_members {
            self.operation.visit_id(ti.id);
            walk_trait_item(self, ti);
        }
    }

    fn visit_impl_item(&mut self, ii: &ImplItem) {
        if !self.skip_members {
            self.operation.visit_id(ii.id);
            walk_impl_item(self, ii);
        }
    }

    fn visit_lifetime(&mut self, lifetime: &Lifetime) {
        self.operation.visit_id(lifetime.id);
    }

    fn visit_lifetime_def(&mut self, def: &LifetimeDef) {
        self.visit_lifetime(&def.lifetime);
    }

    fn visit_trait_ref(&mut self, trait_ref: &TraitRef) {
        self.operation.visit_id(trait_ref.ref_id);
        walk_trait_ref(self, trait_ref);
    }
}

/// Computes the id range for a single fn body, ignoring nested items.
pub fn compute_id_range_for_fn_body(fk: FnKind,
                                    decl: &FnDecl,
                                    body: &Block,
                                    sp: Span,
                                    id: NodeId)
                                    -> IdRange {
    let mut visitor = IdRangeComputingVisitor { result: IdRange::max() };
    let mut id_visitor = IdVisitor::new(&mut visitor);
    id_visitor.visit_fn(fk, decl, body, sp, id);
    id_visitor.operation.result
}
