// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::Abi;
use ast::*;
use ast;
use codemap::Span;
use parse;
use owned_slice::OwnedSlice;

// Context-passing AST walker. Each overridden visit method has full control
// over what happens with its node, it can do its own traversal of the node's
// children (potentially passing in different contexts to each), call
// visit::visit_* to apply the default traversal algorithm (again, it can
// override the context), or prevent deeper traversal by doing nothing.
//
// Note: it is an important invariant that the default visitor walks the body
// of a function in "execution order" (more concretely, reverse post-order
// with respect to the CFG implied by the AST), meaning that if AST node A may
// execute before AST node B, then A is visited first.  The borrow checker in
// particular relies on this property.

pub enum FnKind<'a> {
    // fn foo() or extern "Abi" fn foo()
    FkItemFn(Ident, &'a Generics, FnStyle, Abi),

    // fn foo(&self)
    FkMethod(Ident, &'a Generics, &'a Method),

    // |x, y| ...
    // proc(x, y) ...
    FkFnBlock,
}

pub fn name_of_fn(fk: &FnKind) -> Ident {
    match *fk {
        FkItemFn(name, _, _, _) | FkMethod(name, _, _) => name,
        FkFnBlock(..) => parse::token::special_idents::invalid
    }
}

pub fn generics_of_fn(fk: &FnKind) -> Generics {
    match *fk {
        FkItemFn(_, generics, _, _) |
        FkMethod(_, generics, _) => {
            (*generics).clone()
        }
        FkFnBlock(..) => {
            Generics {
                lifetimes: Vec::new(),
                ty_params: OwnedSlice::empty(),
            }
        }
    }
}

/// Each method of the Visitor trait is a hook to be potentially
/// overriden.  Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g. the `visit_mod` method by default calls `visit::walk_mod`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method.  (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<E: Clone> {

    fn visit_ident(&mut self, _sp: Span, _ident: Ident, _e: E) {
        /*! Visit the idents */
    }
    fn visit_mod(&mut self, m: &Mod, _s: Span, _n: NodeId, e: E) { walk_mod(self, m, e) }
    fn visit_view_item(&mut self, i: &ViewItem, e: E) { walk_view_item(self, i, e) }
    fn visit_foreign_item(&mut self, i: &ForeignItem, e: E) { walk_foreign_item(self, i, e) }
    fn visit_item(&mut self, i: &Item, e: E) { walk_item(self, i, e) }
    fn visit_local(&mut self, l: &Local, e: E) { walk_local(self, l, e) }
    fn visit_block(&mut self, b: &Block, e: E) { walk_block(self, b, e) }
    fn visit_stmt(&mut self, s: &Stmt, e: E) { walk_stmt(self, s, e) }
    fn visit_arm(&mut self, a: &Arm, e: E) { walk_arm(self, a, e) }
    fn visit_pat(&mut self, p: &Pat, e: E) { walk_pat(self, p, e) }
    fn visit_decl(&mut self, d: &Decl, e: E) { walk_decl(self, d, e) }
    fn visit_expr(&mut self, ex: &Expr, e: E) { walk_expr(self, ex, e) }
    fn visit_expr_post(&mut self, _ex: &Expr, _e: E) { }
    fn visit_ty(&mut self, t: &Ty, e: E) { walk_ty(self, t, e) }
    fn visit_generics(&mut self, g: &Generics, e: E) { walk_generics(self, g, e) }
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, _: NodeId, e: E) {
        walk_fn(self, fk, fd, b, s, e)
    }
    fn visit_ty_method(&mut self, t: &TypeMethod, e: E) { walk_ty_method(self, t, e) }
    fn visit_trait_method(&mut self, t: &TraitMethod, e: E) { walk_trait_method(self, t, e) }
    fn visit_struct_def(&mut self, s: &StructDef, _: Ident, _: &Generics, _: NodeId, e: E) {
        walk_struct_def(self, s, e)
    }
    fn visit_struct_field(&mut self, s: &StructField, e: E) { walk_struct_field(self, s, e) }
    fn visit_variant(&mut self, v: &Variant, g: &Generics, e: E) { walk_variant(self, v, g, e) }
    fn visit_opt_lifetime_ref(&mut self,
                              _span: Span,
                              opt_lifetime: &Option<Lifetime>,
                              env: E) {
        /*!
         * Visits an optional reference to a lifetime. The `span` is
         * the span of some surrounding reference should opt_lifetime
         * be None.
         */
        match *opt_lifetime {
            Some(ref l) => self.visit_lifetime_ref(l, env),
            None => ()
        }
    }
    fn visit_lifetime_ref(&mut self, _lifetime: &Lifetime, _e: E) {
        /*! Visits a reference to a lifetime */
    }
    fn visit_lifetime_decl(&mut self, _lifetime: &Lifetime, _e: E) {
        /*! Visits a declaration of a lifetime */
    }
    fn visit_explicit_self(&mut self, es: &ExplicitSelf, e: E) {
        walk_explicit_self(self, es, e)
    }
    fn visit_mac(&mut self, macro: &Mac, e: E) {
        walk_mac(self, macro, e)
    }
    fn visit_path(&mut self, path: &Path, _id: ast::NodeId, e: E) {
        walk_path(self, path, e)
    }
}

pub fn walk_inlined_item<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                  item: &ast::InlinedItem,
                                                  env: E) {
    match *item {
        IIItem(i) => visitor.visit_item(i, env),
        IIForeign(i) => visitor.visit_foreign_item(i, env),
        IIMethod(_, _, m) => walk_method_helper(visitor, m, env),
    }
}


pub fn walk_crate<E: Clone, V: Visitor<E>>(visitor: &mut V, krate: &Crate, env: E) {
    visitor.visit_mod(&krate.module, krate.span, CRATE_NODE_ID, env)
}

pub fn walk_mod<E: Clone, V: Visitor<E>>(visitor: &mut V, module: &Mod, env: E) {
    for view_item in module.view_items.iter() {
        visitor.visit_view_item(view_item, env.clone())
    }

    for item in module.items.iter() {
        visitor.visit_item(*item, env.clone())
    }
}

pub fn walk_view_item<E: Clone, V: Visitor<E>>(visitor: &mut V, vi: &ViewItem, env: E) {
    match vi.node {
        ViewItemExternCrate(name, _, _) => {
            visitor.visit_ident(vi.span, name, env)
        }
        ViewItemUse(ref vp) => {
            match vp.node {
                ViewPathSimple(ident, ref path, id) => {
                    visitor.visit_ident(vp.span, ident, env.clone());
                    visitor.visit_path(path, id, env.clone());
                }
                ViewPathGlob(ref path, id) => {
                    visitor.visit_path(path, id, env.clone());
                }
                ViewPathList(ref path, ref list, _) => {
                    for id in list.iter() {
                        visitor.visit_ident(id.span, id.node.name, env.clone())
                    }
                    walk_path(visitor, path, env.clone());
                }
            }
        }
    }
}

pub fn walk_local<E: Clone, V: Visitor<E>>(visitor: &mut V, local: &Local, env: E) {
    visitor.visit_pat(local.pat, env.clone());
    visitor.visit_ty(local.ty, env.clone());
    match local.init {
        None => {}
        Some(initializer) => visitor.visit_expr(initializer, env),
    }
}

pub fn walk_explicit_self<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                   explicit_self: &ExplicitSelf,
                                                   env: E) {
    match explicit_self.node {
        SelfStatic | SelfValue | SelfUniq => {}
        SelfRegion(ref lifetime, _) => {
            visitor.visit_opt_lifetime_ref(explicit_self.span, lifetime, env)
        }
    }
}

/// Like with walk_method_helper this doesn't correspond to a method
/// in Visitor, and so it gets a _helper suffix.
pub fn walk_trait_ref_helper<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                      trait_ref: &TraitRef,
                                                      env: E) {
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id, env)
}

pub fn walk_item<E: Clone, V: Visitor<E>>(visitor: &mut V, item: &Item, env: E) {
    visitor.visit_ident(item.span, item.ident, env.clone());
    match item.node {
        ItemStatic(typ, _, expr) => {
            visitor.visit_ty(typ, env.clone());
            visitor.visit_expr(expr, env);
        }
        ItemFn(declaration, fn_style, abi, ref generics, body) => {
            visitor.visit_fn(&FkItemFn(item.ident, generics, fn_style, abi),
                             declaration,
                             body,
                             item.span,
                             item.id,
                             env)
        }
        ItemMod(ref module) => {
            visitor.visit_mod(module, item.span, item.id, env)
        }
        ItemForeignMod(ref foreign_module) => {
            for view_item in foreign_module.view_items.iter() {
                visitor.visit_view_item(view_item, env.clone())
            }
            for foreign_item in foreign_module.items.iter() {
                visitor.visit_foreign_item(*foreign_item, env.clone())
            }
        }
        ItemTy(typ, ref type_parameters) => {
            visitor.visit_ty(typ, env.clone());
            visitor.visit_generics(type_parameters, env)
        }
        ItemEnum(ref enum_definition, ref type_parameters) => {
            visitor.visit_generics(type_parameters, env.clone());
            walk_enum_def(visitor, enum_definition, type_parameters, env)
        }
        ItemImpl(ref type_parameters,
                 ref trait_reference,
                 typ,
                 ref methods) => {
            visitor.visit_generics(type_parameters, env.clone());
            match *trait_reference {
                Some(ref trait_reference) => walk_trait_ref_helper(visitor,
                                                                   trait_reference, env.clone()),
                None => ()
            }
            visitor.visit_ty(typ, env.clone());
            for method in methods.iter() {
                walk_method_helper(visitor, *method, env.clone())
            }
        }
        ItemStruct(struct_definition, ref generics) => {
            visitor.visit_generics(generics, env.clone());
            visitor.visit_struct_def(struct_definition,
                                     item.ident,
                                     generics,
                                     item.id,
                                     env)
        }
        ItemTrait(ref generics, _, ref trait_paths, ref methods) => {
            visitor.visit_generics(generics, env.clone());
            for trait_path in trait_paths.iter() {
                visitor.visit_path(&trait_path.path,
                                   trait_path.ref_id,
                                   env.clone())
            }
            for method in methods.iter() {
                visitor.visit_trait_method(method, env.clone())
            }
        }
        ItemMac(ref macro) => visitor.visit_mac(macro, env),
    }
}

pub fn walk_enum_def<E: Clone, V:Visitor<E>>(visitor: &mut V,
                                             enum_definition: &EnumDef,
                                             generics: &Generics,
                                             env: E) {
    for &variant in enum_definition.variants.iter() {
        visitor.visit_variant(variant, generics, env.clone());
    }
}

pub fn walk_variant<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                             variant: &Variant,
                                             generics: &Generics,
                                             env: E) {
    visitor.visit_ident(variant.span, variant.node.name, env.clone());

    match variant.node.kind {
        TupleVariantKind(ref variant_arguments) => {
            for variant_argument in variant_arguments.iter() {
                visitor.visit_ty(variant_argument.ty, env.clone())
            }
        }
        StructVariantKind(struct_definition) => {
            visitor.visit_struct_def(struct_definition,
                                     variant.node.name,
                                     generics,
                                     variant.node.id,
                                     env.clone())
        }
    }
    match variant.node.disr_expr {
        Some(expr) => visitor.visit_expr(expr, env),
        None => ()
    }
}

pub fn skip_ty<E, V: Visitor<E>>(_: &mut V, _: &Ty, _: E) {
    // Empty!
}

pub fn walk_ty<E: Clone, V: Visitor<E>>(visitor: &mut V, typ: &Ty, env: E) {
    match typ.node {
        TyUniq(ty) | TyVec(ty) | TyBox(ty) => {
            visitor.visit_ty(ty, env)
        }
        TyPtr(ref mutable_type) => {
            visitor.visit_ty(mutable_type.ty, env)
        }
        TyRptr(ref lifetime, ref mutable_type) => {
            visitor.visit_opt_lifetime_ref(typ.span, lifetime, env.clone());
            visitor.visit_ty(mutable_type.ty, env)
        }
        TyTup(ref tuple_element_types) => {
            for &tuple_element_type in tuple_element_types.iter() {
                visitor.visit_ty(tuple_element_type, env.clone())
            }
        }
        TyClosure(ref function_declaration, ref region) => {
            for argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(argument.ty, env.clone())
            }
            visitor.visit_ty(function_declaration.decl.output, env.clone());
            for bounds in function_declaration.bounds.iter() {
                walk_ty_param_bounds(visitor, bounds, env.clone())
            }
            visitor.visit_opt_lifetime_ref(
                typ.span,
                region,
                env.clone());
            walk_lifetime_decls(visitor, &function_declaration.lifetimes,
                                env.clone());
        }
        TyProc(ref function_declaration) => {
            for argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(argument.ty, env.clone())
            }
            visitor.visit_ty(function_declaration.decl.output, env.clone());
            for bounds in function_declaration.bounds.iter() {
                walk_ty_param_bounds(visitor, bounds, env.clone())
            }
            walk_lifetime_decls(visitor, &function_declaration.lifetimes,
                                env.clone());
        }
        TyBareFn(ref function_declaration) => {
            for argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(argument.ty, env.clone())
            }
            visitor.visit_ty(function_declaration.decl.output, env.clone());
            walk_lifetime_decls(visitor, &function_declaration.lifetimes,
                                env.clone());
        }
        TyPath(ref path, ref bounds, id) => {
            visitor.visit_path(path, id, env.clone());
            for bounds in bounds.iter() {
                walk_ty_param_bounds(visitor, bounds, env.clone())
            }
        }
        TyFixedLengthVec(ty, expression) => {
            visitor.visit_ty(ty, env.clone());
            visitor.visit_expr(expression, env)
        }
        TyTypeof(expression) => {
            visitor.visit_expr(expression, env)
        }
        TyNil | TyBot | TyInfer => {}
    }
}

fn walk_lifetime_decls<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                lifetimes: &Vec<Lifetime>,
                                                env: E) {
    for l in lifetimes.iter() {
        visitor.visit_lifetime_decl(l, env.clone());
    }
}

pub fn walk_path<E: Clone, V: Visitor<E>>(visitor: &mut V, path: &Path, env: E) {
    for segment in path.segments.iter() {
        visitor.visit_ident(path.span, segment.identifier, env.clone());

        for &typ in segment.types.iter() {
            visitor.visit_ty(typ, env.clone());
        }
        for lifetime in segment.lifetimes.iter() {
            visitor.visit_lifetime_ref(lifetime, env.clone());
        }
    }
}

pub fn walk_pat<E: Clone, V: Visitor<E>>(visitor: &mut V, pattern: &Pat, env: E) {
    match pattern.node {
        PatEnum(ref path, ref children) => {
            visitor.visit_path(path, pattern.id, env.clone());
            for children in children.iter() {
                for child in children.iter() {
                    visitor.visit_pat(*child, env.clone())
                }
            }
        }
        PatStruct(ref path, ref fields, _) => {
            visitor.visit_path(path, pattern.id, env.clone());
            for field in fields.iter() {
                visitor.visit_pat(field.pat, env.clone())
            }
        }
        PatTup(ref tuple_elements) => {
            for tuple_element in tuple_elements.iter() {
                visitor.visit_pat(*tuple_element, env.clone())
            }
        }
        PatUniq(subpattern) |
        PatRegion(subpattern) => {
            visitor.visit_pat(subpattern, env)
        }
        PatIdent(_, ref path, ref optional_subpattern) => {
            visitor.visit_path(path, pattern.id, env.clone());
            match *optional_subpattern {
                None => {}
                Some(subpattern) => visitor.visit_pat(subpattern, env),
            }
        }
        PatLit(expression) => visitor.visit_expr(expression, env),
        PatRange(lower_bound, upper_bound) => {
            visitor.visit_expr(lower_bound, env.clone());
            visitor.visit_expr(upper_bound, env)
        }
        PatWild | PatWildMulti => (),
        PatVec(ref prepattern, ref slice_pattern, ref postpatterns) => {
            for prepattern in prepattern.iter() {
                visitor.visit_pat(*prepattern, env.clone())
            }
            for slice_pattern in slice_pattern.iter() {
                visitor.visit_pat(*slice_pattern, env.clone())
            }
            for postpattern in postpatterns.iter() {
                visitor.visit_pat(*postpattern, env.clone())
            }
        }
    }
}

pub fn walk_foreign_item<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                  foreign_item: &ForeignItem,
                                                  env: E) {
    visitor.visit_ident(foreign_item.span, foreign_item.ident, env.clone());

    match foreign_item.node {
        ForeignItemFn(function_declaration, ref generics) => {
            walk_fn_decl(visitor, function_declaration, env.clone());
            visitor.visit_generics(generics, env)
        }
        ForeignItemStatic(typ, _) => visitor.visit_ty(typ, env),
    }
}

pub fn walk_ty_param_bounds<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                     bounds: &OwnedSlice<TyParamBound>,
                                                     env: E) {
    for bound in bounds.iter() {
        match *bound {
            TraitTyParamBound(ref typ) => {
                walk_trait_ref_helper(visitor, typ, env.clone())
            }
            StaticRegionTyParamBound => {}
            OtherRegionTyParamBound(..) => {}
        }
    }
}

pub fn walk_generics<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                              generics: &Generics,
                                              env: E) {
    for type_parameter in generics.ty_params.iter() {
        walk_ty_param_bounds(visitor, &type_parameter.bounds, env.clone());
        match type_parameter.default {
            Some(ty) => visitor.visit_ty(ty, env.clone()),
            None => {}
        }
    }
    walk_lifetime_decls(visitor, &generics.lifetimes, env);
}

pub fn walk_fn_decl<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                             function_declaration: &FnDecl,
                                             env: E) {
    for argument in function_declaration.inputs.iter() {
        visitor.visit_pat(argument.pat, env.clone());
        visitor.visit_ty(argument.ty, env.clone())
    }
    visitor.visit_ty(function_declaration.output, env)
}

// Note: there is no visit_method() method in the visitor, instead override
// visit_fn() and check for FkMethod().  I named this visit_method_helper()
// because it is not a default impl of any method, though I doubt that really
// clarifies anything. - Niko
pub fn walk_method_helper<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                   method: &Method,
                                                   env: E) {
    visitor.visit_ident(method.span, method.ident, env.clone());
    visitor.visit_fn(&FkMethod(method.ident, &method.generics, method),
                     method.decl,
                     method.body,
                     method.span,
                     method.id,
                     env)
}

pub fn walk_fn<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                        function_kind: &FnKind,
                                        function_declaration: &FnDecl,
                                        function_body: &Block,
                                        _span: Span,
                                        env: E) {
    walk_fn_decl(visitor, function_declaration, env.clone());

    match *function_kind {
        FkItemFn(_, generics, _, _) => {
            visitor.visit_generics(generics, env.clone());
        }
        FkMethod(_, generics, method) => {
            visitor.visit_generics(generics, env.clone());

            visitor.visit_explicit_self(&method.explicit_self, env.clone());
        }
        FkFnBlock(..) => {}
    }

    visitor.visit_block(function_body, env)
}

pub fn walk_ty_method<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                               method_type: &TypeMethod,
                                               env: E) {
    visitor.visit_ident(method_type.span, method_type.ident, env.clone());
    visitor.visit_explicit_self(&method_type.explicit_self, env.clone());
    for argument_type in method_type.decl.inputs.iter() {
        visitor.visit_ty(argument_type.ty, env.clone())
    }
    visitor.visit_generics(&method_type.generics, env.clone());
    visitor.visit_ty(method_type.decl.output, env);
}

pub fn walk_trait_method<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                  trait_method: &TraitMethod,
                                                  env: E) {
    match *trait_method {
        Required(ref method_type) => {
            visitor.visit_ty_method(method_type, env)
        }
        Provided(method) => walk_method_helper(visitor, method, env),
    }
}

pub fn walk_struct_def<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                struct_definition: &StructDef,
                                                env: E) {
    match struct_definition.super_struct {
        Some(t) => visitor.visit_ty(t, env.clone()),
        None => {},
    }
    for field in struct_definition.fields.iter() {
        visitor.visit_struct_field(field, env.clone())
    }
}

pub fn walk_struct_field<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                                  struct_field: &StructField,
                                                  env: E) {
    match struct_field.node.kind {
        NamedField(name, _) => {
            visitor.visit_ident(struct_field.span, name, env.clone())
        }
        _ => {}
    }

    visitor.visit_ty(struct_field.node.ty, env)
}

pub fn walk_block<E: Clone, V: Visitor<E>>(visitor: &mut V, block: &Block, env: E) {
    for view_item in block.view_items.iter() {
        visitor.visit_view_item(view_item, env.clone())
    }
    for statement in block.stmts.iter() {
        visitor.visit_stmt(*statement, env.clone())
    }
    walk_expr_opt(visitor, block.expr, env)
}

pub fn walk_stmt<E: Clone, V: Visitor<E>>(visitor: &mut V, statement: &Stmt, env: E) {
    match statement.node {
        StmtDecl(declaration, _) => visitor.visit_decl(declaration, env),
        StmtExpr(expression, _) | StmtSemi(expression, _) => {
            visitor.visit_expr(expression, env)
        }
        StmtMac(ref macro, _) => visitor.visit_mac(macro, env),
    }
}

pub fn walk_decl<E: Clone, V: Visitor<E>>(visitor: &mut V, declaration: &Decl, env: E) {
    match declaration.node {
        DeclLocal(ref local) => visitor.visit_local(*local, env),
        DeclItem(item) => visitor.visit_item(item, env),
    }
}

pub fn walk_expr_opt<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                              optional_expression: Option<@Expr>,
                                              env: E) {
    match optional_expression {
        None => {}
        Some(expression) => visitor.visit_expr(expression, env),
    }
}

pub fn walk_exprs<E: Clone, V: Visitor<E>>(visitor: &mut V,
                                           expressions: &[@Expr],
                                           env: E) {
    for expression in expressions.iter() {
        visitor.visit_expr(*expression, env.clone())
    }
}

pub fn walk_mac<E, V: Visitor<E>>(_: &mut V, _: &Mac, _: E) {
    // Empty!
}

pub fn walk_expr<E: Clone, V: Visitor<E>>(visitor: &mut V, expression: &Expr, env: E) {
    match expression.node {
        ExprVstore(subexpression, _) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        ExprBox(place, subexpression) => {
            visitor.visit_expr(place, env.clone());
            visitor.visit_expr(subexpression, env.clone())
        }
        ExprVec(ref subexpressions) => {
            walk_exprs(visitor, subexpressions.as_slice(), env.clone())
        }
        ExprRepeat(element, count) => {
            visitor.visit_expr(element, env.clone());
            visitor.visit_expr(count, env.clone())
        }
        ExprStruct(ref path, ref fields, optional_base) => {
            visitor.visit_path(path, expression.id, env.clone());
            for field in fields.iter() {
                visitor.visit_expr(field.expr, env.clone())
            }
            walk_expr_opt(visitor, optional_base, env.clone())
        }
        ExprTup(ref subexpressions) => {
            for subexpression in subexpressions.iter() {
                visitor.visit_expr(*subexpression, env.clone())
            }
        }
        ExprCall(callee_expression, ref arguments) => {
            for argument in arguments.iter() {
                visitor.visit_expr(*argument, env.clone())
            }
            visitor.visit_expr(callee_expression, env.clone())
        }
        ExprMethodCall(_, ref types, ref arguments) => {
            walk_exprs(visitor, arguments.as_slice(), env.clone());
            for &typ in types.iter() {
                visitor.visit_ty(typ, env.clone())
            }
        }
        ExprBinary(_, left_expression, right_expression) => {
            visitor.visit_expr(left_expression, env.clone());
            visitor.visit_expr(right_expression, env.clone())
        }
        ExprAddrOf(_, subexpression) | ExprUnary(_, subexpression) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        ExprLit(_) => {}
        ExprCast(subexpression, typ) => {
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_ty(typ, env.clone())
        }
        ExprIf(head_expression, if_block, optional_else) => {
            visitor.visit_expr(head_expression, env.clone());
            visitor.visit_block(if_block, env.clone());
            walk_expr_opt(visitor, optional_else, env.clone())
        }
        ExprWhile(subexpression, block) => {
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_block(block, env.clone())
        }
        ExprForLoop(pattern, subexpression, block, _) => {
            visitor.visit_pat(pattern, env.clone());
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_block(block, env.clone())
        }
        ExprLoop(block, _) => visitor.visit_block(block, env.clone()),
        ExprMatch(subexpression, ref arms) => {
            visitor.visit_expr(subexpression, env.clone());
            for arm in arms.iter() {
                visitor.visit_arm(arm, env.clone())
            }
        }
        ExprFnBlock(function_declaration, body) => {
            visitor.visit_fn(&FkFnBlock,
                             function_declaration,
                             body,
                             expression.span,
                             expression.id,
                             env.clone())
        }
        ExprProc(function_declaration, body) => {
            visitor.visit_fn(&FkFnBlock,
                             function_declaration,
                             body,
                             expression.span,
                             expression.id,
                             env.clone())
        }
        ExprBlock(block) => visitor.visit_block(block, env.clone()),
        ExprAssign(left_hand_expression, right_hand_expression) => {
            visitor.visit_expr(right_hand_expression, env.clone());
            visitor.visit_expr(left_hand_expression, env.clone())
        }
        ExprAssignOp(_, left_expression, right_expression) => {
            visitor.visit_expr(right_expression, env.clone());
            visitor.visit_expr(left_expression, env.clone())
        }
        ExprField(subexpression, _, ref types) => {
            visitor.visit_expr(subexpression, env.clone());
            for &typ in types.iter() {
                visitor.visit_ty(typ, env.clone())
            }
        }
        ExprIndex(main_expression, index_expression) => {
            visitor.visit_expr(main_expression, env.clone());
            visitor.visit_expr(index_expression, env.clone())
        }
        ExprPath(ref path) => {
            visitor.visit_path(path, expression.id, env.clone())
        }
        ExprBreak(_) | ExprAgain(_) => {}
        ExprRet(optional_expression) => {
            walk_expr_opt(visitor, optional_expression, env.clone())
        }
        ExprMac(ref macro) => visitor.visit_mac(macro, env.clone()),
        ExprParen(subexpression) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        ExprInlineAsm(ref assembler) => {
            for &(_, input) in assembler.inputs.iter() {
                visitor.visit_expr(input, env.clone())
            }
            for &(_, output) in assembler.outputs.iter() {
                visitor.visit_expr(output, env.clone())
            }
        }
    }

    visitor.visit_expr_post(expression, env.clone())
}

pub fn walk_arm<E: Clone, V: Visitor<E>>(visitor: &mut V, arm: &Arm, env: E) {
    for pattern in arm.pats.iter() {
        visitor.visit_pat(*pattern, env.clone())
    }
    walk_expr_opt(visitor, arm.guard, env.clone());
    visitor.visit_expr(arm.body, env)
}
