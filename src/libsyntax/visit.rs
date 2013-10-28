// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::AbiSet;
use ast::*;
use ast;
use codemap::Span;
use parse;
use opt_vec;
use opt_vec::OptVec;

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

pub enum fn_kind<'self> {
    // fn foo() or extern "Abi" fn foo()
    fk_item_fn(Ident, &'self Generics, purity, AbiSet),

    // fn foo(&self)
    fk_method(Ident, &'self Generics, &'self method),

    // @fn(x, y) { ... }
    fk_anon(ast::Sigil),

    // |x, y| ...
    fk_fn_block,
}

pub fn name_of_fn(fk: &fn_kind) -> Ident {
    match *fk {
      fk_item_fn(name, _, _, _) | fk_method(name, _, _) => {
          name
      }
      fk_anon(*) | fk_fn_block(*) => parse::token::special_idents::anon,
    }
}

pub fn generics_of_fn(fk: &fn_kind) -> Generics {
    match *fk {
        fk_item_fn(_, generics, _, _) |
        fk_method(_, generics, _) => {
            (*generics).clone()
        }
        fk_anon(*) | fk_fn_block(*) => {
            Generics {
                lifetimes: opt_vec::Empty,
                ty_params: opt_vec::Empty,
            }
        }
    }
}

pub trait Visitor<E:Clone> {
    fn visit_mod(&mut self, m:&_mod, _s:Span, _n:NodeId, e:E) { walk_mod(self, m, e) }
    fn visit_view_item(&mut self, i:&view_item, e:E) { walk_view_item(self, i, e) }
    fn visit_foreign_item(&mut self, i:@foreign_item, e:E) { walk_foreign_item(self, i, e) }
    fn visit_item(&mut self, i:@item, e:E) { walk_item(self, i, e) }
    fn visit_local(&mut self, l:@Local, e:E) { walk_local(self, l, e) }
    fn visit_block(&mut self, b:&Block, e:E) { walk_block(self, b, e) }
    fn visit_stmt(&mut self, s:@Stmt, e:E) { walk_stmt(self, s, e) }
    fn visit_arm(&mut self, a:&Arm, e:E) { walk_arm(self, a, e) }
    fn visit_pat(&mut self, p:@Pat, e:E) { walk_pat(self, p, e) }
    fn visit_decl(&mut self, d:@Decl, e:E) { walk_decl(self, d, e) }
    fn visit_expr(&mut self, ex:@Expr, e:E) { walk_expr(self, ex, e) }
    fn visit_expr_post(&mut self, _ex:@Expr, _e:E) { }
    fn visit_ty(&mut self, _t:&Ty, _e:E) { }
    fn visit_generics(&mut self, g:&Generics, e:E) { walk_generics(self, g, e) }
    fn visit_fn(&mut self, fk:&fn_kind, fd:&fn_decl, b:&Block, s:Span, n:NodeId, e:E) {
        walk_fn(self, fk, fd, b, s, n , e)
    }
    fn visit_ty_method(&mut self, t:&TypeMethod, e:E) { walk_ty_method(self, t, e) }
    fn visit_trait_method(&mut self, t:&trait_method, e:E) { walk_trait_method(self, t, e) }
    fn visit_struct_def(&mut self, s:@struct_def, i:Ident, g:&Generics, n:NodeId, e:E) {
        walk_struct_def(self, s, i, g, n, e)
    }
    fn visit_struct_field(&mut self, s:@struct_field, e:E) { walk_struct_field(self, s, e) }
    fn visit_mac(&mut self, m:&mac, e:E) { walk_mac(self, m, e); }
}

pub fn walk_crate<E:Clone, V:Visitor<E>>(visitor: &mut V, crate: &Crate, env: E) {
    visitor.visit_mod(&crate.module, crate.span, CRATE_NODE_ID, env)
}

pub fn walk_mod<E:Clone, V:Visitor<E>>(visitor: &mut V, module: &_mod, env: E) {
    for view_item in module.view_items.iter() {
        visitor.visit_view_item(view_item, env.clone())
    }
    for item in module.items.iter() {
        visitor.visit_item(*item, env.clone())
    }
}

pub fn walk_view_item<E:Clone, V:Visitor<E>>(_: &mut V, _: &view_item, _: E) {
    // Empty!
}

pub fn walk_local<E:Clone, V:Visitor<E>>(visitor: &mut V, local: &Local, env: E) {
    visitor.visit_pat(local.pat, env.clone());
    visitor.visit_ty(&local.ty, env.clone());
    match local.init {
        None => {}
        Some(initializer) => visitor.visit_expr(initializer, env),
    }
}

fn walk_trait_ref<E:Clone, V:Visitor<E>>(visitor: &mut V,
                            trait_ref: &ast::trait_ref,
                            env: E) {
    walk_path(visitor, &trait_ref.path, env)
}

pub fn walk_item<E:Clone, V:Visitor<E>>(visitor: &mut V, item: &item, env: E) {
    match item.node {
        item_static(ref typ, _, expr) => {
            visitor.visit_ty(typ, env.clone());
            visitor.visit_expr(expr, env);
        }
        item_fn(ref declaration, purity, abi, ref generics, ref body) => {
            visitor.visit_fn(&fk_item_fn(item.ident, generics, purity, abi),
                             declaration,
                             body,
                             item.span,
                             item.id,
                             env)
        }
        item_mod(ref module) => {
            visitor.visit_mod(module, item.span, item.id, env)
        }
        item_foreign_mod(ref foreign_module) => {
            for view_item in foreign_module.view_items.iter() {
                visitor.visit_view_item(view_item, env.clone())
            }
            for foreign_item in foreign_module.items.iter() {
                visitor.visit_foreign_item(*foreign_item, env.clone())
            }
        }
        item_ty(ref typ, ref type_parameters) => {
            visitor.visit_ty(typ, env.clone());
            visitor.visit_generics(type_parameters, env)
        }
        item_enum(ref enum_definition, ref type_parameters) => {
            visitor.visit_generics(type_parameters, env.clone());
            walk_enum_def(visitor, enum_definition, type_parameters, env)
        }
        item_impl(ref type_parameters,
                  ref trait_references,
                  ref typ,
                  ref methods) => {
            visitor.visit_generics(type_parameters, env.clone());
            for trait_reference in trait_references.iter() {
                walk_trait_ref(visitor, trait_reference, env.clone())
            }
            visitor.visit_ty(typ, env.clone());
            for method in methods.iter() {
                walk_method_helper(visitor, *method, env.clone())
            }
        }
        item_struct(struct_definition, ref generics) => {
            visitor.visit_generics(generics, env.clone());
            visitor.visit_struct_def(struct_definition,
                                     item.ident,
                                     generics,
                                     item.id,
                                     env)
        }
        item_trait(ref generics, ref trait_paths, ref methods) => {
            visitor.visit_generics(generics, env.clone());
            for trait_path in trait_paths.iter() {
                walk_path(visitor, &trait_path.path, env.clone())
            }
            for method in methods.iter() {
                visitor.visit_trait_method(method, env.clone())
            }
        }
        item_mac(ref macro) => visitor.visit_mac(macro, env),
    }
}

pub fn walk_enum_def<E:Clone, V:Visitor<E>>(visitor: &mut V,
                               enum_definition: &ast::enum_def,
                               generics: &Generics,
                               env: E) {
    for variant in enum_definition.variants.iter() {
        match variant.node.kind {
            tuple_variant_kind(ref variant_arguments) => {
                for variant_argument in variant_arguments.iter() {
                    visitor.visit_ty(&variant_argument.ty, env.clone())
                }
            }
            struct_variant_kind(struct_definition) => {
                visitor.visit_struct_def(struct_definition,
                                         variant.node.name,
                                         generics,
                                         variant.node.id,
                                         env.clone())
            }
        }
    }
}

pub fn skip_ty<E, V:Visitor<E>>(_: &mut V, _: &Ty, _: E) {
    // Empty!
}

pub fn walk_ty<E:Clone, V:Visitor<E>>(visitor: &mut V, typ: &Ty, env: E) {
    match typ.node {
        ty_box(ref mutable_type) | ty_uniq(ref mutable_type) |
        ty_vec(ref mutable_type) | ty_ptr(ref mutable_type) |
        ty_rptr(_, ref mutable_type) => {
            visitor.visit_ty(mutable_type.ty, env)
        }
        ty_tup(ref tuple_element_types) => {
            for tuple_element_type in tuple_element_types.iter() {
                visitor.visit_ty(tuple_element_type, env.clone())
            }
        }
        ty_closure(ref function_declaration) => {
             for argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(&argument.ty, env.clone())
             }
             visitor.visit_ty(&function_declaration.decl.output, env.clone());
             for bounds in function_declaration.bounds.iter() {
                walk_ty_param_bounds(visitor, bounds, env.clone())
             }
        }
        ty_bare_fn(ref function_declaration) => {
            for argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(&argument.ty, env.clone())
            }
            visitor.visit_ty(&function_declaration.decl.output, env.clone())
        }
        ty_path(ref path, ref bounds, _) => {
            walk_path(visitor, path, env.clone());
            for bounds in bounds.iter() {
                walk_ty_param_bounds(visitor, bounds, env.clone())
            }
        }
        ty_fixed_length_vec(ref mutable_type, expression) => {
            visitor.visit_ty(mutable_type.ty, env.clone());
            visitor.visit_expr(expression, env)
        }
        ty_typeof(expression) => {
            visitor.visit_expr(expression, env)
        }
        ty_nil | ty_bot | ty_mac(_) | ty_infer => ()
    }
}

pub fn walk_path<E:Clone, V:Visitor<E>>(visitor: &mut V, path: &Path, env: E) {
    for segment in path.segments.iter() {
        for typ in segment.types.iter() {
            visitor.visit_ty(typ, env.clone())
        }
    }
}

pub fn walk_pat<E:Clone, V:Visitor<E>>(visitor: &mut V, pattern: &Pat, env: E) {
    match pattern.node {
        PatEnum(ref path, ref children) => {
            walk_path(visitor, path, env.clone());
            for children in children.iter() {
                for child in children.iter() {
                    visitor.visit_pat(*child, env.clone())
                }
            }
        }
        PatStruct(ref path, ref fields, _) => {
            walk_path(visitor, path, env.clone());
            for field in fields.iter() {
                visitor.visit_pat(field.pat, env.clone())
            }
        }
        PatTup(ref tuple_elements) => {
            for tuple_element in tuple_elements.iter() {
                visitor.visit_pat(*tuple_element, env.clone())
            }
        }
        PatBox(subpattern) |
        PatUniq(subpattern) |
        PatRegion(subpattern) => {
            visitor.visit_pat(subpattern, env)
        }
        PatIdent(_, ref path, ref optional_subpattern) => {
            walk_path(visitor, path, env.clone());
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
        PatWild => (),
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

pub fn walk_foreign_item<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                   foreign_item: &foreign_item,
                                   env: E) {
    match foreign_item.node {
        foreign_item_fn(ref function_declaration, ref generics) => {
            walk_fn_decl(visitor, function_declaration, env.clone());
            visitor.visit_generics(generics, env)
        }
        foreign_item_static(ref typ, _) => visitor.visit_ty(typ, env),
    }
}

pub fn walk_ty_param_bounds<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                      bounds: &OptVec<TyParamBound>,
                                      env: E) {
    for bound in bounds.iter() {
        match *bound {
            TraitTyParamBound(ref typ) => {
                walk_trait_ref(visitor, typ, env.clone())
            }
            RegionTyParamBound => {}
        }
    }
}

pub fn walk_generics<E:Clone, V:Visitor<E>>(visitor: &mut V,
                               generics: &Generics,
                               env: E) {
    for type_parameter in generics.ty_params.iter() {
        walk_ty_param_bounds(visitor, &type_parameter.bounds, env.clone())
    }
}

pub fn walk_fn_decl<E:Clone, V:Visitor<E>>(visitor: &mut V,
                              function_declaration: &fn_decl,
                              env: E) {
    for argument in function_declaration.inputs.iter() {
        visitor.visit_pat(argument.pat, env.clone());
        visitor.visit_ty(&argument.ty, env.clone())
    }
    visitor.visit_ty(&function_declaration.output, env)
}

// Note: there is no visit_method() method in the visitor, instead override
// visit_fn() and check for fk_method().  I named this visit_method_helper()
// because it is not a default impl of any method, though I doubt that really
// clarifies anything. - Niko
pub fn walk_method_helper<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                    method: &method,
                                    env: E) {
    visitor.visit_fn(&fk_method(method.ident, &method.generics, method),
                     &method.decl,
                     &method.body,
                     method.span,
                     method.id,
                     env)
}

pub fn walk_fn<E:Clone, V:Visitor<E>>(visitor: &mut V,
                         function_kind: &fn_kind,
                         function_declaration: &fn_decl,
                         function_body: &Block,
                         _: Span,
                         _: NodeId,
                         env: E) {
    walk_fn_decl(visitor, function_declaration, env.clone());
    let generics = generics_of_fn(function_kind);
    visitor.visit_generics(&generics, env.clone());
    visitor.visit_block(function_body, env)
}

pub fn walk_ty_method<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                method_type: &TypeMethod,
                                env: E) {
    for argument_type in method_type.decl.inputs.iter() {
        visitor.visit_ty(&argument_type.ty, env.clone())
    }
    visitor.visit_generics(&method_type.generics, env.clone());
    visitor.visit_ty(&method_type.decl.output, env.clone())
}

pub fn walk_trait_method<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                   trait_method: &trait_method,
                                   env: E) {
    match *trait_method {
        required(ref method_type) => {
            visitor.visit_ty_method(method_type, env)
        }
        provided(method) => walk_method_helper(visitor, method, env),
    }
}

pub fn walk_struct_def<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                 struct_definition: @struct_def,
                                 _: ast::Ident,
                                 _: &Generics,
                                 _: NodeId,
                                 env: E) {
    for field in struct_definition.fields.iter() {
        visitor.visit_struct_field(*field, env.clone())
    }
}

pub fn walk_struct_field<E:Clone, V:Visitor<E>>(visitor: &mut V,
                                   struct_field: &struct_field,
                                   env: E) {
    visitor.visit_ty(&struct_field.node.ty, env)
}

pub fn walk_block<E:Clone, V:Visitor<E>>(visitor: &mut V, block: &Block, env: E) {
    for view_item in block.view_items.iter() {
        visitor.visit_view_item(view_item, env.clone())
    }
    for statement in block.stmts.iter() {
        visitor.visit_stmt(*statement, env.clone())
    }
    walk_expr_opt(visitor, block.expr, env)
}

pub fn walk_stmt<E:Clone, V:Visitor<E>>(visitor: &mut V, statement: &Stmt, env: E) {
    match statement.node {
        StmtDecl(declaration, _) => visitor.visit_decl(declaration, env),
        StmtExpr(expression, _) | StmtSemi(expression, _) => {
            visitor.visit_expr(expression, env)
        }
        StmtMac(ref macro, _) => visitor.visit_mac(macro, env),
    }
}

pub fn walk_decl<E:Clone, V:Visitor<E>>(visitor: &mut V, declaration: &Decl, env: E) {
    match declaration.node {
        DeclLocal(ref local) => visitor.visit_local(*local, env),
        DeclItem(item) => visitor.visit_item(item, env),
    }
}

pub fn walk_expr_opt<E:Clone, V:Visitor<E>>(visitor: &mut V,
                         optional_expression: Option<@Expr>,
                         env: E) {
    match optional_expression {
        None => {}
        Some(expression) => visitor.visit_expr(expression, env),
    }
}

pub fn walk_exprs<E:Clone, V:Visitor<E>>(visitor: &mut V,
                            expressions: &[@Expr],
                            env: E) {
    for expression in expressions.iter() {
        visitor.visit_expr(*expression, env.clone())
    }
}

pub fn walk_mac<E, V:Visitor<E>>(_: &mut V, _: &mac, _: E) {
    // Empty!
}

pub fn walk_expr<E:Clone, V:Visitor<E>>(visitor: &mut V, expression: @Expr, env: E) {
    match expression.node {
        ExprVstore(subexpression, _) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        ExprVec(ref subexpressions, _) => {
            walk_exprs(visitor, *subexpressions, env.clone())
        }
        ExprRepeat(element, count, _) => {
            visitor.visit_expr(element, env.clone());
            visitor.visit_expr(count, env.clone())
        }
        ExprStruct(ref path, ref fields, optional_base) => {
            walk_path(visitor, path, env.clone());
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
        ExprCall(callee_expression, ref arguments, _) => {
            for argument in arguments.iter() {
                visitor.visit_expr(*argument, env.clone())
            }
            visitor.visit_expr(callee_expression, env.clone())
        }
        ExprMethodCall(_, callee, _, ref types, ref arguments, _) => {
            walk_exprs(visitor, *arguments, env.clone());
            for typ in types.iter() {
                visitor.visit_ty(typ, env.clone())
            }
            visitor.visit_expr(callee, env.clone())
        }
        ExprBinary(_, _, left_expression, right_expression) => {
            visitor.visit_expr(left_expression, env.clone());
            visitor.visit_expr(right_expression, env.clone())
        }
        ExprAddrOf(_, subexpression) |
        ExprUnary(_, _, subexpression) |
        ExprDoBody(subexpression) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        ExprLit(_) => {}
        ExprCast(subexpression, ref typ) => {
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_ty(typ, env.clone())
        }
        ExprIf(head_expression, ref if_block, optional_else) => {
            visitor.visit_expr(head_expression, env.clone());
            visitor.visit_block(if_block, env.clone());
            walk_expr_opt(visitor, optional_else, env.clone())
        }
        ExprWhile(subexpression, ref block) => {
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_block(block, env.clone())
        }
        ExprForLoop(pattern, subexpression, ref block, _) => {
            visitor.visit_pat(pattern, env.clone());
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_block(block, env.clone())
        }
        ExprLoop(ref block, _) => visitor.visit_block(block, env.clone()),
        ExprMatch(subexpression, ref arms) => {
            visitor.visit_expr(subexpression, env.clone());
            for arm in arms.iter() {
                visitor.visit_arm(arm, env.clone())
            }
        }
        ExprFnBlock(ref function_declaration, ref body) => {
            visitor.visit_fn(&fk_fn_block,
                             function_declaration,
                             body,
                             expression.span,
                             expression.id,
                             env.clone())
        }
        ExprProc(ref function_declaration, ref body) => {
            visitor.visit_fn(&fk_fn_block,
                             function_declaration,
                             body,
                             expression.span,
                             expression.id,
                             env.clone())
        }
        ExprBlock(ref block) => visitor.visit_block(block, env.clone()),
        ExprAssign(left_hand_expression, right_hand_expression) => {
            visitor.visit_expr(right_hand_expression, env.clone());
            visitor.visit_expr(left_hand_expression, env.clone())
        }
        ExprAssignOp(_, _, left_expression, right_expression) => {
            visitor.visit_expr(right_expression, env.clone());
            visitor.visit_expr(left_expression, env.clone())
        }
        ExprField(subexpression, _, ref types) => {
            visitor.visit_expr(subexpression, env.clone());
            for typ in types.iter() {
                visitor.visit_ty(typ, env.clone())
            }
        }
        ExprIndex(_, main_expression, index_expression) => {
            visitor.visit_expr(main_expression, env.clone());
            visitor.visit_expr(index_expression, env.clone())
        }
        ExprPath(ref path) => walk_path(visitor, path, env.clone()),
        ExprSelf | ExprBreak(_) | ExprAgain(_) => {}
        ExprRet(optional_expression) => {
            walk_expr_opt(visitor, optional_expression, env.clone())
        }
        ExprLogLevel => {}
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

pub fn walk_arm<E:Clone, V:Visitor<E>>(visitor: &mut V, arm: &Arm, env: E) {
    for pattern in arm.pats.iter() {
        visitor.visit_pat(*pattern, env.clone())
    }
    walk_expr_opt(visitor, arm.guard, env.clone());
    visitor.visit_block(&arm.body, env)
}
