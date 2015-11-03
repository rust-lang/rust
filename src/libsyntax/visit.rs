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
use codemap::Span;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FnKind<'a> {
    /// fn foo() or extern "Abi" fn foo()
    ItemFn(Ident, &'a Generics, Unsafety, Constness, Abi, Visibility),

    /// fn foo(&self)
    Method(Ident, &'a MethodSig, Option<Visibility>),

    /// |x, y| {}
    Closure,
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
pub trait Visitor<'v> : Sized {
    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, span: Span, ident: Ident) {
        walk_ident(self, span, ident);
    }
    fn visit_mod(&mut self, m: &'v Mod, _s: Span, _n: NodeId) { walk_mod(self, m) }
    fn visit_foreign_item(&mut self, i: &'v ForeignItem) { walk_foreign_item(self, i) }
    fn visit_item(&mut self, i: &'v Item) { walk_item(self, i) }
    fn visit_local(&mut self, l: &'v Local) { walk_local(self, l) }
    fn visit_block(&mut self, b: &'v Block) { walk_block(self, b) }
    fn visit_stmt(&mut self, s: &'v Stmt) { walk_stmt(self, s) }
    fn visit_arm(&mut self, a: &'v Arm) { walk_arm(self, a) }
    fn visit_pat(&mut self, p: &'v Pat) { walk_pat(self, p) }
    fn visit_decl(&mut self, d: &'v Decl) { walk_decl(self, d) }
    fn visit_expr(&mut self, ex: &'v Expr) { walk_expr(self, ex) }
    fn visit_expr_post(&mut self, _ex: &'v Expr) { }
    fn visit_ty(&mut self, t: &'v Ty) { walk_ty(self, t) }
    fn visit_generics(&mut self, g: &'v Generics) { walk_generics(self, g) }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: &'v Block, s: Span, _: NodeId) {
        walk_fn(self, fk, fd, b, s)
    }
    fn visit_trait_item(&mut self, ti: &'v TraitItem) { walk_trait_item(self, ti) }
    fn visit_impl_item(&mut self, ii: &'v ImplItem) { walk_impl_item(self, ii) }
    fn visit_trait_ref(&mut self, t: &'v TraitRef) { walk_trait_ref(self, t) }
    fn visit_ty_param_bound(&mut self, bounds: &'v TyParamBound) {
        walk_ty_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'v PolyTraitRef, m: &'v TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self, s: &'v VariantData, _: Ident,
                        _: &'v Generics, _: NodeId, _: Span) {
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'v StructField) { walk_struct_field(self, s) }
    fn visit_enum_def(&mut self, enum_definition: &'v EnumDef,
                      generics: &'v Generics, item_id: NodeId, _: Span) {
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
    fn visit_mac(&mut self, _mac: &'v Mac) {
        panic!("visit_mac disabled by default");
        // NB: see note about macros above.
        // if you really want a visitor that
        // works on macros, use this
        // definition in your trait impl:
        // visit::walk_mac(self, _mac)
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
    fn visit_attribute(&mut self, _attr: &'v Attribute) {}
    fn visit_macro_def(&mut self, macro_def: &'v MacroDef) {
        walk_macro_def(self, macro_def)
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

pub fn walk_crate<'v, V: Visitor<'v>>(visitor: &mut V, krate: &'v Crate) {
    visitor.visit_mod(&krate.module, krate.span, CRATE_NODE_ID);
    walk_list!(visitor, visit_attribute, &krate.attrs);
    walk_list!(visitor, visit_macro_def, &krate.exported_macros);
}

pub fn walk_macro_def<'v, V: Visitor<'v>>(visitor: &mut V, macro_def: &'v MacroDef) {
    visitor.visit_ident(macro_def.span, macro_def.ident);
    walk_opt_ident(visitor, macro_def.span, macro_def.imported_from);
    walk_list!(visitor, visit_attribute, &macro_def.attrs);
}

pub fn walk_mod<'v, V: Visitor<'v>>(visitor: &mut V, module: &'v Mod) {
    walk_list!(visitor, visit_item, &module.items);
}

pub fn walk_local<'v, V: Visitor<'v>>(visitor: &mut V, local: &'v Local) {
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
    walk_list!(visitor, visit_expr, &local.init);
}

pub fn walk_lifetime<'v, V: Visitor<'v>>(visitor: &mut V, lifetime: &'v Lifetime) {
    visitor.visit_name(lifetime.span, lifetime.name);
}

pub fn walk_lifetime_def<'v, V: Visitor<'v>>(visitor: &mut V,
                                              lifetime_def: &'v LifetimeDef) {
    visitor.visit_lifetime(&lifetime_def.lifetime);
    walk_list!(visitor, visit_lifetime, &lifetime_def.bounds);
}

pub fn walk_explicit_self<'v, V: Visitor<'v>>(visitor: &mut V,
                                              explicit_self: &'v ExplicitSelf) {
    match explicit_self.node {
        SelfStatic => {},
        SelfValue(ident) => {
            visitor.visit_ident(explicit_self.span, ident)
        }
        SelfRegion(ref opt_lifetime, _, ident) => {
            visitor.visit_ident(explicit_self.span, ident);
            walk_list!(visitor, visit_lifetime, opt_lifetime);
        }
        SelfExplicit(ref typ, ident) => {
            visitor.visit_ident(explicit_self.span, ident);
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

pub fn walk_trait_ref<'v,V>(visitor: &mut V,
                                   trait_ref: &'v TraitRef)
    where V: Visitor<'v>
{
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<'v, V: Visitor<'v>>(visitor: &mut V, item: &'v Item) {
    visitor.visit_ident(item.span, item.ident);
    match item.node {
        ItemExternCrate(opt_name) => {
            walk_opt_name(visitor, item.span, opt_name)
        }
        ItemUse(ref vp) => {
            match vp.node {
                ViewPathSimple(ident, ref path) => {
                    visitor.visit_ident(vp.span, ident);
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
            visitor.visit_fn(FnKind::ItemFn(item.ident, generics, unsafety,
                                            constness, abi, item.vis),
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
        ItemImpl(_, _,
                 ref type_parameters,
                 ref opt_trait_reference,
                 ref typ,
                 ref impl_items) => {
            visitor.visit_generics(type_parameters);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_impl_item, impl_items);
        }
        ItemStruct(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition, item.ident,
                                     generics, item.id, item.span);
        }
        ItemTrait(_, ref generics, ref bounds, ref methods) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_trait_item, methods);
        }
        ItemMac(ref mac) => visitor.visit_mac(mac),
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'v, V: Visitor<'v>>(visitor: &mut V,
                                         enum_definition: &'v EnumDef,
                                         generics: &'v Generics,
                                         item_id: NodeId) {
    walk_list!(visitor, visit_variant, &enum_definition.variants, generics, item_id);
}

pub fn walk_variant<'v, V: Visitor<'v>>(visitor: &mut V,
                                        variant: &'v Variant,
                                        generics: &'v Generics,
                                        item_id: NodeId) {
    visitor.visit_ident(variant.span, variant.node.name);
    visitor.visit_variant_data(&variant.node.data, variant.node.name,
                             generics, item_id, variant.span);
    walk_list!(visitor, visit_expr, &variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.node.attrs);
}

pub fn walk_ty<'v, V: Visitor<'v>>(visitor: &mut V, typ: &'v Ty) {
    match typ.node {
        TyVec(ref ty) | TyParen(ref ty) => {
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
        TyMac(ref mac) => {
            visitor.visit_mac(mac)
        }
    }
}

pub fn walk_path<'v, V: Visitor<'v>>(visitor: &mut V, path: &'v Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_path_list_item<'v, V: Visitor<'v>>(visitor: &mut V, prefix: &'v Path,
                                               item: &'v PathListItem) {
    for segment in &prefix.segments {
        visitor.visit_path_segment(prefix.span, segment);
    }

    walk_opt_ident(visitor, item.span, item.node.name());
    walk_opt_ident(visitor, item.span, item.node.rename());
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
    visitor.visit_ident(type_binding.span, type_binding.ident);
    visitor.visit_ty(&type_binding.ty);
}

pub fn walk_pat<'v, V: Visitor<'v>>(visitor: &mut V, pattern: &'v Pat) {
    match pattern.node {
        PatEnum(ref path, ref opt_children) => {
            visitor.visit_path(path, pattern.id);
            if let Some(ref children) = *opt_children {
                walk_list!(visitor, visit_pat, children);
            }
        }
        PatQPath(ref qself, ref path) => {
            visitor.visit_ty(&qself.ty);
            visitor.visit_path(path, pattern.id)
        }
        PatStruct(ref path, ref fields, _) => {
            visitor.visit_path(path, pattern.id);
            for field in fields {
                visitor.visit_ident(field.span, field.node.ident);
                visitor.visit_pat(&field.node.pat)
            }
        }
        PatTup(ref tuple_elements) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatBox(ref subpattern) |
        PatRegion(ref subpattern, _) => {
            visitor.visit_pat(subpattern)
        }
        PatIdent(_, ref pth1, ref optional_subpattern) => {
            visitor.visit_ident(pth1.span, pth1.node);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatLit(ref expression) => visitor.visit_expr(expression),
        PatRange(ref lower_bound, ref upper_bound) => {
            visitor.visit_expr(lower_bound);
            visitor.visit_expr(upper_bound)
        }
        PatWild => (),
        PatVec(ref prepatterns, ref slice_pattern, ref postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
        PatMac(ref mac) => visitor.visit_mac(mac),
    }
}

pub fn walk_foreign_item<'v, V: Visitor<'v>>(visitor: &mut V,
                                             foreign_item: &'v ForeignItem) {
    visitor.visit_ident(foreign_item.span, foreign_item.ident);

    match foreign_item.node {
        ForeignItemFn(ref function_declaration, ref generics) => {
            walk_fn_decl(visitor, function_declaration);
            visitor.visit_generics(generics)
        }
        ForeignItemStatic(ref typ, _) => visitor.visit_ty(typ),
    }

    walk_list!(visitor, visit_attribute, &foreign_item.attrs);
}

pub fn walk_ty_param_bound<'v, V: Visitor<'v>>(visitor: &mut V,
                                               bound: &'v TyParamBound) {
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
        visitor.visit_ident(param.span, param.ident);
        walk_list!(visitor, visit_ty_param_bound, &param.bounds);
        walk_list!(visitor, visit_ty, &param.default);
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
            WherePredicate::EqPredicate(WhereEqPredicate{id,
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

pub fn walk_fn_kind<'v, V: Visitor<'v>>(visitor: &mut V,
                                        function_kind: FnKind<'v>) {
    match function_kind {
        FnKind::ItemFn(_, generics, _, _, _, _) => {
            visitor.visit_generics(generics);
        }
        FnKind::Method(_, sig, _) => {
            visitor.visit_generics(&sig.generics);
            visitor.visit_explicit_self(&sig.explicit_self);
        }
        FnKind::Closure => {}
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
    visitor.visit_ident(trait_item.span, trait_item.ident);
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
            visitor.visit_fn(FnKind::Method(trait_item.ident, sig, None), &sig.decl,
                             body, trait_item.span, trait_item.id);
        }
        TypeTraitItem(ref bounds, ref default) => {
            walk_list!(visitor, visit_ty_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
    }
}

pub fn walk_impl_item<'v, V: Visitor<'v>>(visitor: &mut V, impl_item: &'v ImplItem) {
    visitor.visit_ident(impl_item.span, impl_item.ident);
    walk_list!(visitor, visit_attribute, &impl_item.attrs);
    match impl_item.node {
        ImplItemKind::Const(ref ty, ref expr) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expr);
        }
        ImplItemKind::Method(ref sig, ref body) => {
            visitor.visit_fn(FnKind::Method(impl_item.ident, sig, Some(impl_item.vis)), &sig.decl,
                             body, impl_item.span, impl_item.id);
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_ty(ty);
        }
        ImplItemKind::Macro(ref mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_struct_def<'v, V: Visitor<'v>>(visitor: &mut V,
                                           struct_definition: &'v VariantData) {
    walk_list!(visitor, visit_struct_field, struct_definition.fields());
}

pub fn walk_struct_field<'v, V: Visitor<'v>>(visitor: &mut V,
                                             struct_field: &'v StructField) {
    walk_opt_ident(visitor, struct_field.span, struct_field.node.ident());
    visitor.visit_ty(&struct_field.node.ty);
    walk_list!(visitor, visit_attribute, &struct_field.node.attrs);
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
        StmtMac(ref mac, _, ref attrs) => {
            visitor.visit_mac(mac);
            for attr in attrs.as_attrs() {
                visitor.visit_attribute(attr);
            }
        }
    }
}

pub fn walk_decl<'v, V: Visitor<'v>>(visitor: &mut V, declaration: &'v Decl) {
    match declaration.node {
        DeclLocal(ref local) => visitor.visit_local(local),
        DeclItem(ref item) => visitor.visit_item(item),
    }
}

pub fn walk_mac<'v, V: Visitor<'v>>(_: &mut V, _: &'v Mac) {
    // Empty!
}

pub fn walk_expr<'v, V: Visitor<'v>>(visitor: &mut V, expression: &'v Expr) {
    match expression.node {
        ExprBox(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprInPlace(ref place, ref subexpression) => {
            visitor.visit_expr(place);
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
                visitor.visit_ident(field.ident.span, field.ident.node);
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
        ExprMethodCall(ref ident, ref types, ref arguments) => {
            visitor.visit_ident(ident.span, ident.node);
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
        ExprCast(ref subexpression, ref typ) => {
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
        ExprIfLet(ref pattern, ref subexpression, ref if_block, ref optional_else) => {
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprWhileLet(ref pattern, ref subexpression, ref block, opt_ident) => {
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_ident(visitor, expression.span, opt_ident)
        }
        ExprForLoop(ref pattern, ref subexpression, ref block, opt_ident) => {
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
            walk_opt_ident(visitor, expression.span, opt_ident)
        }
        ExprLoop(ref block, opt_ident) => {
            visitor.visit_block(block);
            walk_opt_ident(visitor, expression.span, opt_ident)
        }
        ExprMatch(ref subexpression, ref arms) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprClosure(_, ref function_declaration, ref body) => {
            visitor.visit_fn(FnKind::Closure,
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
        ExprField(ref subexpression, ref ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ident(ident.span, ident.node);
        }
        ExprTupField(ref subexpression, _) => {
            visitor.visit_expr(subexpression);
        }
        ExprIndex(ref main_expression, ref index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprRange(ref start, ref end) => {
            walk_list!(visitor, visit_expr, start);
            walk_list!(visitor, visit_expr, end);
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
        ExprMac(ref mac) => visitor.visit_mac(mac),
        ExprParen(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprInlineAsm(ref ia) => {
            for &(_, ref input) in &ia.inputs {
                visitor.visit_expr(&input)
            }
            for &(_, ref output, _) in &ia.outputs {
                visitor.visit_expr(&output)
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
