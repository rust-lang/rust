// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! AST walker. Each overridden visit method has full control over what
//! happens with its node, it can do its own traversal of the node's children,
//! call `visit::walk_*` to apply the default traversal algorithm, or prevent
//! deeper traversal by doing nothing.
//!
//! Note: it is an important invariant that the default visitor walks the body
//! of a function in "execution order" (more concretely, reverse post-order
//! with respect to the CFG implied by the AST), meaning that if AST node A may
//! execute before AST node B, then A is visited first.  The borrow checker in
//! particular relies on this property.
//!
//! Note: walking an AST before macro expansion is probably a bad idea. For
//! instance, a walker looking for item names in a module will miss all of
//! those that are created by the expansion of a macro.

use abi::Abi;
use ast::*;
use syntax_pos::Span;
use codemap::Spanned;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FnKind<'a> {
    /// fn foo() or extern "Abi" fn foo()
    ItemFn(Ident, &'a Generics, Unsafety, Spanned<Constness>, Abi, &'a Visibility, &'a Block),

    /// fn foo(&self)
    Method(Ident, &'a MethodSig, Option<&'a Visibility>, &'a Block),

    /// |x, y| body
    Closure(&'a Expr),
}

/// Each method of the Visitor trait is a hook to be potentially
/// overridden.  Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g. the `visit_mod` method by default calls `visit::walk_mod`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method.  (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<'ast>: Sized {
    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, span: Span, ident: Ident) {
        walk_ident(self, span, ident);
    }
    fn visit_mod(&mut self, m: &'ast Mod, _s: Span, _n: NodeId) { walk_mod(self, m) }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) { walk_foreign_item(self, i) }
    fn visit_item(&mut self, i: &'ast Item) { walk_item(self, i) }
    fn visit_local(&mut self, l: &'ast Local) { walk_local(self, l) }
    fn visit_block(&mut self, b: &'ast Block) { walk_block(self, b) }
    fn visit_stmt(&mut self, s: &'ast Stmt) { walk_stmt(self, s) }
    fn visit_arm(&mut self, a: &'ast Arm) { walk_arm(self, a) }
    fn visit_pat(&mut self, p: &'ast Pat) { walk_pat(self, p) }
    fn visit_expr(&mut self, ex: &'ast Expr) { walk_expr(self, ex) }
    fn visit_expr_post(&mut self, _ex: &'ast Expr) { }
    fn visit_ty(&mut self, t: &'ast Ty) { walk_ty(self, t) }
    fn visit_generics(&mut self, g: &'ast Generics) { walk_generics(self, g) }
    fn visit_fn(&mut self, fk: FnKind<'ast>, fd: &'ast FnDecl, s: Span, _: NodeId) {
        walk_fn(self, fk, fd, s)
    }
    fn visit_trait_item(&mut self, ti: &'ast TraitItem) { walk_trait_item(self, ti) }
    fn visit_impl_item(&mut self, ii: &'ast ImplItem) { walk_impl_item(self, ii) }
    fn visit_trait_ref(&mut self, t: &'ast TraitRef) { walk_trait_ref(self, t) }
    fn visit_ty_param_bound(&mut self, bounds: &'ast TyParamBound) {
        walk_ty_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef, m: &'ast TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self, s: &'ast VariantData, _: Ident,
                          _: &'ast Generics, _: NodeId, _: Span) {
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'ast StructField) { walk_struct_field(self, s) }
    fn visit_enum_def(&mut self, enum_definition: &'ast EnumDef,
                      generics: &'ast Generics, item_id: NodeId, _: Span) {
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(&mut self, v: &'ast Variant, g: &'ast Generics, item_id: NodeId) {
        walk_variant(self, v, g, item_id)
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        walk_lifetime(self, lifetime)
    }
    fn visit_lifetime_def(&mut self, lifetime: &'ast LifetimeDef) {
        walk_lifetime_def(self, lifetime)
    }
    fn visit_mac(&mut self, _mac: &'ast Mac) {
        panic!("visit_mac disabled by default");
        // NB: see note about macros above.
        // if you really want a visitor that
        // works on macros, use this
        // definition in your trait impl:
        // visit::walk_mac(self, _mac)
    }
    fn visit_path(&mut self, path: &'ast Path, _id: NodeId) {
        walk_path(self, path)
    }
    fn visit_path_list_item(&mut self, prefix: &'ast Path, item: &'ast PathListItem) {
        walk_path_list_item(self, prefix, item)
    }
    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'ast PathSegment) {
        walk_path_segment(self, path_span, path_segment)
    }
    fn visit_path_parameters(&mut self, path_span: Span, path_parameters: &'ast PathParameters) {
        walk_path_parameters(self, path_span, path_parameters)
    }
    fn visit_assoc_type_binding(&mut self, type_binding: &'ast TypeBinding) {
        walk_assoc_type_binding(self, type_binding)
    }
    fn visit_attribute(&mut self, _attr: &'ast Attribute) {}
    fn visit_macro_def(&mut self, macro_def: &'ast MacroDef) {
        walk_macro_def(self, macro_def)
    }
    fn visit_vis(&mut self, vis: &'ast Visibility) {
        walk_vis(self, vis)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'ast FunctionRetTy) {
        walk_fn_ret_ty(self, ret_ty)
    }
}

#[macro_export]
macro_rules! walk_list {
    ($visitor: expr, $method: ident, $list: expr) => {
        for elem in $list {
            $visitor.$method(elem)
        }
    };
    ($visitor: expr, $method: ident, $list: expr, $($extra_args: expr),*) => {
        for elem in $list {
            $visitor.$method(elem, $($extra_args,)*)
        }
    }
}

pub fn walk_opt_name<'a, V: Visitor<'a>>(visitor: &mut V, span: Span, opt_name: Option<Name>) {
    if let Some(name) = opt_name {
        visitor.visit_name(span, name);
    }
}

pub fn walk_opt_ident<'a, V: Visitor<'a>>(visitor: &mut V, span: Span, opt_ident: Option<Ident>) {
    if let Some(ident) = opt_ident {
        visitor.visit_ident(span, ident);
    }
}

pub fn walk_opt_sp_ident<'a, V: Visitor<'a>>(visitor: &mut V,
                                             opt_sp_ident: &Option<Spanned<Ident>>) {
    if let Some(ref sp_ident) = *opt_sp_ident {
        visitor.visit_ident(sp_ident.span, sp_ident.node);
    }
}

pub fn walk_ident<'a, V: Visitor<'a>>(visitor: &mut V, span: Span, ident: Ident) {
    visitor.visit_name(span, ident.name);
}

pub fn walk_crate<'a, V: Visitor<'a>>(visitor: &mut V, krate: &'a Crate) {
    visitor.visit_mod(&krate.module, krate.span, CRATE_NODE_ID);
    walk_list!(visitor, visit_attribute, &krate.attrs);
    walk_list!(visitor, visit_macro_def, &krate.exported_macros);
}

pub fn walk_macro_def<'a, V: Visitor<'a>>(visitor: &mut V, macro_def: &'a MacroDef) {
    visitor.visit_ident(macro_def.span, macro_def.ident);
    walk_list!(visitor, visit_attribute, &macro_def.attrs);
}

pub fn walk_mod<'a, V: Visitor<'a>>(visitor: &mut V, module: &'a Mod) {
    walk_list!(visitor, visit_item, &module.items);
}

pub fn walk_local<'a, V: Visitor<'a>>(visitor: &mut V, local: &'a Local) {
    for attr in local.attrs.iter() {
        visitor.visit_attribute(attr);
    }
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
    walk_list!(visitor, visit_expr, &local.init);
}

pub fn walk_lifetime<'a, V: Visitor<'a>>(visitor: &mut V, lifetime: &'a Lifetime) {
    visitor.visit_name(lifetime.span, lifetime.name);
}

pub fn walk_lifetime_def<'a, V: Visitor<'a>>(visitor: &mut V, lifetime_def: &'a LifetimeDef) {
    visitor.visit_lifetime(&lifetime_def.lifetime);
    walk_list!(visitor, visit_lifetime, &lifetime_def.bounds);
    walk_list!(visitor, visit_attribute, &*lifetime_def.attrs);
}

pub fn walk_poly_trait_ref<'a, V>(visitor: &mut V,
                                  trait_ref: &'a PolyTraitRef,
                                  _: &TraitBoundModifier)
    where V: Visitor<'a>,
{
    walk_list!(visitor, visit_lifetime_def, &trait_ref.bound_lifetimes);
    visitor.visit_trait_ref(&trait_ref.trait_ref);
}

pub fn walk_trait_ref<'a, V: Visitor<'a>>(visitor: &mut V, trait_ref: &'a TraitRef) {
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a Item) {
    visitor.visit_vis(&item.vis);
    visitor.visit_ident(item.span, item.ident);
    match item.node {
        ItemKind::ExternCrate(opt_name) => {
            walk_opt_name(visitor, item.span, opt_name)
        }
        ItemKind::Use(ref vp) => {
            match vp.node {
                ViewPathSimple(ident, ref path) => {
                    visitor.visit_ident(vp.span, ident);
                    visitor.visit_path(path, item.id);
                }
                ViewPathGlob(ref path) => {
                    visitor.visit_path(path, item.id);
                }
                ViewPathList(ref prefix, ref list) => {
                    visitor.visit_path(prefix, item.id);
                    for item in list {
                        visitor.visit_path_list_item(prefix, item)
                    }
                }
            }
        }
        ItemKind::Static(ref typ, _, ref expr) |
        ItemKind::Const(ref typ, ref expr) => {
            visitor.visit_ty(typ);
            visitor.visit_expr(expr);
        }
        ItemKind::Fn(ref declaration, unsafety, constness, abi, ref generics, ref body) => {
            visitor.visit_fn(FnKind::ItemFn(item.ident, generics, unsafety,
                                            constness, abi, &item.vis, body),
                             declaration,
                             item.span,
                             item.id)
        }
        ItemKind::Mod(ref module) => {
            visitor.visit_mod(module, item.span, item.id)
        }
        ItemKind::ForeignMod(ref foreign_module) => {
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemKind::Ty(ref typ, ref type_parameters) => {
            visitor.visit_ty(typ);
            visitor.visit_generics(type_parameters)
        }
        ItemKind::Enum(ref enum_definition, ref type_parameters) => {
            visitor.visit_generics(type_parameters);
            visitor.visit_enum_def(enum_definition, type_parameters, item.id, item.span)
        }
        ItemKind::DefaultImpl(_, ref trait_ref) => {
            visitor.visit_trait_ref(trait_ref)
        }
        ItemKind::Impl(_, _,
                 ref type_parameters,
                 ref opt_trait_reference,
                 ref typ,
                 ref impl_items) => {
            visitor.visit_generics(type_parameters);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_impl_item, impl_items);
        }
        ItemKind::Struct(ref struct_definition, ref generics) |
        ItemKind::Union(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition, item.ident,
                                     generics, item.id, item.span);
        }
        ItemKind::Trait(_, ref generics, ref bounds, ref methods) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_trait_item, methods);
        }
        ItemKind::Mac(ref mac) => visitor.visit_mac(mac),
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'a, V: Visitor<'a>>(visitor: &mut V,
                                 enum_definition: &'a EnumDef,
                                 generics: &'a Generics,
                                 item_id: NodeId) {
    walk_list!(visitor, visit_variant, &enum_definition.variants, generics, item_id);
}

pub fn walk_variant<'a, V>(visitor: &mut V,
                           variant: &'a Variant,
                           generics: &'a Generics,
                           item_id: NodeId)
    where V: Visitor<'a>,
{
    visitor.visit_ident(variant.span, variant.node.name);
    visitor.visit_variant_data(&variant.node.data, variant.node.name,
                             generics, item_id, variant.span);
    walk_list!(visitor, visit_expr, &variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.node.attrs);
}

pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) {
    match typ.node {
        TyKind::Slice(ref ty) | TyKind::Paren(ref ty) => {
            visitor.visit_ty(ty)
        }
        TyKind::Ptr(ref mutable_type) => {
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Rptr(ref opt_lifetime, ref mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Never => {},
        TyKind::Tup(ref tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(ref function_declaration) => {
            walk_fn_decl(visitor, &function_declaration.decl);
            walk_list!(visitor, visit_lifetime_def, &function_declaration.lifetimes);
        }
        TyKind::Path(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, typ.id);
        }
        TyKind::Array(ref ty, ref expression) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expression)
        }
        TyKind::TraitObject(ref bounds) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
        }
        TyKind::ImplTrait(ref bounds) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
        }
        TyKind::Typeof(ref expression) => {
            visitor.visit_expr(expression)
        }
        TyKind::Infer | TyKind::ImplicitSelf => {}
        TyKind::Mac(ref mac) => {
            visitor.visit_mac(mac)
        }
    }
}

pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_path_list_item<'a, V: Visitor<'a>>(visitor: &mut V,
                                               _prefix: &Path,
                                               item: &'a PathListItem) {
    visitor.visit_ident(item.span, item.node.name);
    walk_opt_ident(visitor, item.span, item.node.rename);
}

pub fn walk_path_segment<'a, V: Visitor<'a>>(visitor: &mut V,
                                             path_span: Span,
                                             segment: &'a PathSegment) {
    visitor.visit_ident(path_span, segment.identifier);
    if let Some(ref parameters) = segment.parameters {
        visitor.visit_path_parameters(path_span, parameters);
    }
}

pub fn walk_path_parameters<'a, V>(visitor: &mut V,
                                   _path_span: Span,
                                   path_parameters: &'a PathParameters)
    where V: Visitor<'a>,
{
    match *path_parameters {
        PathParameters::AngleBracketed(ref data) => {
            walk_list!(visitor, visit_ty, &data.types);
            walk_list!(visitor, visit_lifetime, &data.lifetimes);
            walk_list!(visitor, visit_assoc_type_binding, &data.bindings);
        }
        PathParameters::Parenthesized(ref data) => {
            walk_list!(visitor, visit_ty, &data.inputs);
            walk_list!(visitor, visit_ty, &data.output);
        }
    }
}

pub fn walk_assoc_type_binding<'a, V: Visitor<'a>>(visitor: &mut V,
                                                   type_binding: &'a TypeBinding) {
    visitor.visit_ident(type_binding.span, type_binding.ident);
    visitor.visit_ty(&type_binding.ty);
}

pub fn walk_pat<'a, V: Visitor<'a>>(visitor: &mut V, pattern: &'a Pat) {
    match pattern.node {
        PatKind::TupleStruct(ref path, ref children, _) => {
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref opt_qself, ref path) => {
            if let Some(ref qself) = *opt_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, pattern.id)
        }
        PatKind::Struct(ref path, ref fields, _) => {
            visitor.visit_path(path, pattern.id);
            for field in fields {
                walk_list!(visitor, visit_attribute, field.node.attrs.iter());
                visitor.visit_ident(field.span, field.node.ident);
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
        PatKind::Slice(ref prepatterns, ref slice_pattern, ref postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
        PatKind::Mac(ref mac) => visitor.visit_mac(mac),
    }
}

pub fn walk_foreign_item<'a, V: Visitor<'a>>(visitor: &mut V, foreign_item: &'a ForeignItem) {
    visitor.visit_vis(&foreign_item.vis);
    visitor.visit_ident(foreign_item.span, foreign_item.ident);

    match foreign_item.node {
        ForeignItemKind::Fn(ref function_declaration, ref generics) => {
            walk_fn_decl(visitor, function_declaration);
            visitor.visit_generics(generics)
        }
        ForeignItemKind::Static(ref typ, _) => visitor.visit_ty(typ),
    }

    walk_list!(visitor, visit_attribute, &foreign_item.attrs);
}

pub fn walk_ty_param_bound<'a, V: Visitor<'a>>(visitor: &mut V, bound: &'a TyParamBound) {
    match *bound {
        TraitTyParamBound(ref typ, ref modifier) => {
            visitor.visit_poly_trait_ref(typ, modifier);
        }
        RegionTyParamBound(ref lifetime) => {
            visitor.visit_lifetime(lifetime);
        }
    }
}

pub fn walk_generics<'a, V: Visitor<'a>>(visitor: &mut V, generics: &'a Generics) {
    for param in &generics.ty_params {
        visitor.visit_ident(param.span, param.ident);
        walk_list!(visitor, visit_ty_param_bound, &param.bounds);
        walk_list!(visitor, visit_ty, &param.default);
        walk_list!(visitor, visit_attribute, &*param.attrs);
    }
    walk_list!(visitor, visit_lifetime_def, &generics.lifetimes);
    for predicate in &generics.where_clause.predicates {
        match *predicate {
            WherePredicate::BoundPredicate(WhereBoundPredicate{ref bounded_ty,
                                                               ref bounds,
                                                               ref bound_lifetimes,
                                                               ..}) => {
                visitor.visit_ty(bounded_ty);
                walk_list!(visitor, visit_ty_param_bound, bounds);
                walk_list!(visitor, visit_lifetime_def, bound_lifetimes);
            }
            WherePredicate::RegionPredicate(WhereRegionPredicate{ref lifetime,
                                                                 ref bounds,
                                                                 ..}) => {
                visitor.visit_lifetime(lifetime);
                walk_list!(visitor, visit_lifetime, bounds);
            }
            WherePredicate::EqPredicate(WhereEqPredicate{ref lhs_ty,
                                                         ref rhs_ty,
                                                         ..}) => {
                visitor.visit_ty(lhs_ty);
                visitor.visit_ty(rhs_ty);
            }
        }
    }
}

pub fn walk_fn_ret_ty<'a, V: Visitor<'a>>(visitor: &mut V, ret_ty: &'a FunctionRetTy) {
    if let FunctionRetTy::Ty(ref output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<'a, V: Visitor<'a>>(visitor: &mut V, function_declaration: &'a FnDecl) {
    for argument in &function_declaration.inputs {
        visitor.visit_pat(&argument.pat);
        visitor.visit_ty(&argument.ty)
    }
    visitor.visit_fn_ret_ty(&function_declaration.output)
}

pub fn walk_fn<'a, V>(visitor: &mut V, kind: FnKind<'a>, declaration: &'a FnDecl, _span: Span)
    where V: Visitor<'a>,
{
    match kind {
        FnKind::ItemFn(_, generics, _, _, _, _, body) => {
            visitor.visit_generics(generics);
            walk_fn_decl(visitor, declaration);
            visitor.visit_block(body);
        }
        FnKind::Method(_, ref sig, _, body) => {
            visitor.visit_generics(&sig.generics);
            walk_fn_decl(visitor, declaration);
            visitor.visit_block(body);
        }
        FnKind::Closure(body) => {
            walk_fn_decl(visitor, declaration);
            visitor.visit_expr(body);
        }
    }
}

pub fn walk_trait_item<'a, V: Visitor<'a>>(visitor: &mut V, trait_item: &'a TraitItem) {
    visitor.visit_ident(trait_item.span, trait_item.ident);
    walk_list!(visitor, visit_attribute, &trait_item.attrs);
    match trait_item.node {
        TraitItemKind::Const(ref ty, ref default) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, default);
        }
        TraitItemKind::Method(ref sig, None) => {
            visitor.visit_generics(&sig.generics);
            walk_fn_decl(visitor, &sig.decl);
        }
        TraitItemKind::Method(ref sig, Some(ref body)) => {
            visitor.visit_fn(FnKind::Method(trait_item.ident, sig, None, body),
                             &sig.decl, trait_item.span, trait_item.id);
        }
        TraitItemKind::Type(ref bounds, ref default) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
        TraitItemKind::Macro(ref mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_impl_item<'a, V: Visitor<'a>>(visitor: &mut V, impl_item: &'a ImplItem) {
    visitor.visit_vis(&impl_item.vis);
    visitor.visit_ident(impl_item.span, impl_item.ident);
    walk_list!(visitor, visit_attribute, &impl_item.attrs);
    match impl_item.node {
        ImplItemKind::Const(ref ty, ref expr) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expr);
        }
        ImplItemKind::Method(ref sig, ref body) => {
            visitor.visit_fn(FnKind::Method(impl_item.ident, sig, Some(&impl_item.vis), body),
                             &sig.decl, impl_item.span, impl_item.id);
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_ty(ty);
        }
        ImplItemKind::Macro(ref mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_struct_def<'a, V: Visitor<'a>>(visitor: &mut V, struct_definition: &'a VariantData) {
    walk_list!(visitor, visit_struct_field, struct_definition.fields());
}

pub fn walk_struct_field<'a, V: Visitor<'a>>(visitor: &mut V, struct_field: &'a StructField) {
    visitor.visit_vis(&struct_field.vis);
    walk_opt_ident(visitor, struct_field.span, struct_field.ident);
    visitor.visit_ty(&struct_field.ty);
    walk_list!(visitor, visit_attribute, &struct_field.attrs);
}

pub fn walk_block<'a, V: Visitor<'a>>(visitor: &mut V, block: &'a Block) {
    walk_list!(visitor, visit_stmt, &block.stmts);
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) {
    match statement.node {
        StmtKind::Local(ref local) => visitor.visit_local(local),
        StmtKind::Item(ref item) => visitor.visit_item(item),
        StmtKind::Expr(ref expression) | StmtKind::Semi(ref expression) => {
            visitor.visit_expr(expression)
        }
        StmtKind::Mac(ref mac) => {
            let (ref mac, _, ref attrs) = **mac;
            visitor.visit_mac(mac);
            for attr in attrs.iter() {
                visitor.visit_attribute(attr);
            }
        }
    }
}

pub fn walk_mac<'a, V: Visitor<'a>>(_: &mut V, _: &Mac) {
    // Empty!
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) {
    for attr in expression.attrs.iter() {
        visitor.visit_attribute(attr);
    }
    match expression.node {
        ExprKind::Box(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::InPlace(ref place, ref subexpression) => {
            visitor.visit_expr(place);
            visitor.visit_expr(subexpression)
        }
        ExprKind::Array(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Repeat(ref element, ref count) => {
            visitor.visit_expr(element);
            visitor.visit_expr(count)
        }
        ExprKind::Struct(ref path, ref fields, ref optional_base) => {
            visitor.visit_path(path, expression.id);
            for field in fields {
                walk_list!(visitor, visit_attribute, field.attrs.iter());
                visitor.visit_ident(field.ident.span, field.ident.node);
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
        ExprKind::MethodCall(ref ident, ref types, ref arguments) => {
            visitor.visit_ident(ident.span, ident.node);
            walk_list!(visitor, visit_ty, types);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, ref subexpression) | ExprKind::Unary(_, ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Lit(_) => {}
        ExprKind::Cast(ref subexpression, ref typ) | ExprKind::Type(ref subexpression, ref typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::If(ref head_expression, ref if_block, ref optional_else) => {
            visitor.visit_expr(head_expression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprKind::While(ref subexpression, ref block, ref opt_sp_ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_sp_ident(visitor, opt_sp_ident);
        }
        ExprKind::IfLet(ref pattern, ref subexpression, ref if_block, ref optional_else) => {
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprKind::WhileLet(ref pattern, ref subexpression, ref block, ref opt_sp_ident) => {
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_sp_ident(visitor, opt_sp_ident);
        }
        ExprKind::ForLoop(ref pattern, ref subexpression, ref block, ref opt_sp_ident) => {
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_sp_ident(visitor, opt_sp_ident);
        }
        ExprKind::Loop(ref block, ref opt_sp_ident) => {
            visitor.visit_block(block);
            walk_opt_sp_ident(visitor, opt_sp_ident);
        }
        ExprKind::Match(ref subexpression, ref arms) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(_, ref function_declaration, ref body, _decl_span) => {
            visitor.visit_fn(FnKind::Closure(body),
                             function_declaration,
                             expression.span,
                             expression.id)
        }
        ExprKind::Block(ref block) => visitor.visit_block(block),
        ExprKind::Assign(ref left_hand_expression, ref right_hand_expression) => {
            visitor.visit_expr(left_hand_expression);
            visitor.visit_expr(right_hand_expression);
        }
        ExprKind::AssignOp(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression);
        }
        ExprKind::Field(ref subexpression, ref ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ident(ident.span, ident.node);
        }
        ExprKind::TupField(ref subexpression, _) => {
            visitor.visit_expr(subexpression);
        }
        ExprKind::Index(ref main_expression, ref index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprKind::Range(ref start, ref end, _) => {
            walk_list!(visitor, visit_expr, start);
            walk_list!(visitor, visit_expr, end);
        }
        ExprKind::Path(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, expression.id)
        }
        ExprKind::Break(ref opt_sp_ident, ref opt_expr) => {
            walk_opt_sp_ident(visitor, opt_sp_ident);
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(ref opt_sp_ident) => {
            walk_opt_sp_ident(visitor, opt_sp_ident);
        }
        ExprKind::Ret(ref optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Mac(ref mac) => visitor.visit_mac(mac),
        ExprKind::Paren(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::InlineAsm(ref ia) => {
            for &(_, ref input) in &ia.inputs {
                visitor.visit_expr(&input)
            }
            for output in &ia.outputs {
                visitor.visit_expr(&output.expr)
            }
        }
        ExprKind::Try(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
    }

    visitor.visit_expr_post(expression)
}

pub fn walk_arm<'a, V: Visitor<'a>>(visitor: &mut V, arm: &'a Arm) {
    walk_list!(visitor, visit_pat, &arm.pats);
    walk_list!(visitor, visit_expr, &arm.guard);
    visitor.visit_expr(&arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
}

pub fn walk_vis<'a, V: Visitor<'a>>(visitor: &mut V, vis: &'a Visibility) {
    if let Visibility::Restricted { ref path, id } = *vis {
        visitor.visit_path(path, id);
    }
}
