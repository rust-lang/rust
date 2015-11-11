// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Simply gives a rought count of the number of nodes in an AST.

use visit::*;
use ast::*;
use codemap::Span;

pub struct NodeCounter {
    pub count: usize,
}

impl NodeCounter {
    pub fn new() -> NodeCounter {
        NodeCounter {
            count: 0,
        }
    }
}

impl<'v> Visitor<'v> for NodeCounter {
    fn visit_ident(&mut self, span: Span, ident: Ident) {
        self.count += 1;
        walk_ident(self, span, ident);
    }
    fn visit_mod(&mut self, m: &'v Mod, _s: Span, _n: NodeId) {
        self.count += 1;
        walk_mod(self, m)
    }
    fn visit_foreign_item(&mut self, i: &'v ForeignItem) {
        self.count += 1;
        walk_foreign_item(self, i)
    }
    fn visit_item(&mut self, i: &'v Item) {
        self.count += 1;
        walk_item(self, i)
    }
    fn visit_local(&mut self, l: &'v Local) {
        self.count += 1;
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'v Block) {
        self.count += 1;
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'v Stmt) {
        self.count += 1;
        walk_stmt(self, s)
    }
    fn visit_arm(&mut self, a: &'v Arm) {
        self.count += 1;
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'v Pat) {
        self.count += 1;
        walk_pat(self, p)
    }
    fn visit_decl(&mut self, d: &'v Decl) {
        self.count += 1;
        walk_decl(self, d)
    }
    fn visit_expr(&mut self, ex: &'v Expr) {
        self.count += 1;
        walk_expr(self, ex)
    }
    fn visit_ty(&mut self, t: &'v Ty) {
        self.count += 1;
        walk_ty(self, t)
    }
    fn visit_generics(&mut self, g: &'v Generics) {
        self.count += 1;
        walk_generics(self, g)
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: &'v Block, s: Span, _: NodeId) {
        self.count += 1;
        walk_fn(self, fk, fd, b, s)
    }
    fn visit_trait_item(&mut self, ti: &'v TraitItem) {
        self.count += 1;
        walk_trait_item(self, ti)
    }
    fn visit_impl_item(&mut self, ii: &'v ImplItem) {
        self.count += 1;
        walk_impl_item(self, ii)
    }
    fn visit_trait_ref(&mut self, t: &'v TraitRef) {
        self.count += 1;
        walk_trait_ref(self, t)
    }
    fn visit_ty_param_bound(&mut self, bounds: &'v TyParamBound) {
        self.count += 1;
        walk_ty_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'v PolyTraitRef, m: &'v TraitBoundModifier) {
        self.count += 1;
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self, s: &'v VariantData, _: Ident,
                        _: &'v Generics, _: NodeId, _: Span) {
        self.count += 1;
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'v StructField) {
        self.count += 1;
        walk_struct_field(self, s)
    }
    fn visit_enum_def(&mut self, enum_definition: &'v EnumDef,
                      generics: &'v Generics, item_id: NodeId, _: Span) {
        self.count += 1;
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(&mut self, v: &'v Variant, g: &'v Generics, item_id: NodeId) {
        self.count += 1;
        walk_variant(self, v, g, item_id)
    }
    fn visit_lifetime(&mut self, lifetime: &'v Lifetime) {
        self.count += 1;
        walk_lifetime(self, lifetime)
    }
    fn visit_lifetime_def(&mut self, lifetime: &'v LifetimeDef) {
        self.count += 1;
        walk_lifetime_def(self, lifetime)
    }
    fn visit_explicit_self(&mut self, es: &'v ExplicitSelf) {
        self.count += 1;
        walk_explicit_self(self, es)
    }
    fn visit_mac(&mut self, _mac: &'v Mac) {
        self.count += 1;
        walk_mac(self, _mac)
    }
    fn visit_path(&mut self, path: &'v Path, _id: NodeId) {
        self.count += 1;
        walk_path(self, path)
    }
    fn visit_path_list_item(&mut self, prefix: &'v Path, item: &'v PathListItem) {
        self.count += 1;
        walk_path_list_item(self, prefix, item)
    }
    fn visit_path_parameters(&mut self, path_span: Span, path_parameters: &'v PathParameters) {
        self.count += 1;
        walk_path_parameters(self, path_span, path_parameters)
    }
    fn visit_assoc_type_binding(&mut self, type_binding: &'v TypeBinding) {
        self.count += 1;
        walk_assoc_type_binding(self, type_binding)
    }
    fn visit_attribute(&mut self, _attr: &'v Attribute) {
        self.count += 1;
    }
    fn visit_macro_def(&mut self, macro_def: &'v MacroDef) {
        self.count += 1;
        walk_macro_def(self, macro_def)
    }

}
