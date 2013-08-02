// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use codemap::span;
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
    fk_item_fn(ident, &'self Generics, purity, AbiSet),

    // fn foo(&self)
    fk_method(ident, &'self Generics, &'self method),

    // @fn(x, y) { ... }
    fk_anon(ast::Sigil),

    // |x, y| ...
    fk_fn_block,
}

pub fn name_of_fn(fk: &fn_kind) -> ident {
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

pub trait Visitor<E> {
    fn visit_mod(@mut self, &_mod, span, NodeId, E);
    fn visit_view_item(@mut self, &view_item, E);
    fn visit_foreign_item(@mut self, @foreign_item, E);
    fn visit_item(@mut self, @item, E);
    fn visit_local(@mut self, @Local, E);
    fn visit_block(@mut self, &Block, E);
    fn visit_stmt(@mut self, @stmt, E);
    fn visit_arm(@mut self, &arm, E);
    fn visit_pat(@mut self, @pat, E);
    fn visit_decl(@mut self, @decl, E);
    fn visit_expr(@mut self, @expr, E);
    fn visit_expr_post(@mut self, @expr, E);
    fn visit_ty(@mut self, &Ty, E);
    fn visit_generics(@mut self, &Generics, E);
    fn visit_fn(@mut self, &fn_kind, &fn_decl, &Block, span, NodeId, E);
    fn visit_ty_method(@mut self, &TypeMethod, E);
    fn visit_trait_method(@mut self, &trait_method, E);
    fn visit_struct_def(@mut self, @struct_def, ident, &Generics, NodeId, E);
    fn visit_struct_field(@mut self, @struct_field, E);
}

pub fn visit_crate<E:Clone>(visitor: @Visitor<E>, crate: &Crate, env: E) {
    visitor.visit_mod(&crate.module, crate.span, CRATE_NODE_ID, env)
}

pub fn visit_mod<E:Clone>(visitor: @Visitor<E>, module: &_mod, env: E) {
    foreach view_item in module.view_items.iter() {
        visitor.visit_view_item(view_item, env.clone())
    }
    foreach item in module.items.iter() {
        visitor.visit_item(*item, env.clone())
    }
}

pub fn visit_view_item<E:Clone>(_: @Visitor<E>, _: &view_item, _: E) {
    // Empty!
}

pub fn visit_local<E:Clone>(visitor: @Visitor<E>, local: &Local, env: E) {
    visitor.visit_pat(local.pat, env.clone());
    visitor.visit_ty(&local.ty, env.clone());
    match local.init {
        None => {}
        Some(initializer) => visitor.visit_expr(initializer, env),
    }
}

fn visit_trait_ref<E:Clone>(visitor: @Visitor<E>,
                            trait_ref: &ast::trait_ref,
                            env: E) {
    visit_path(visitor, &trait_ref.path, env)
}

pub fn visit_item<E:Clone>(visitor: @Visitor<E>, item: &item, env: E) {
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
            foreach view_item in foreign_module.view_items.iter() {
                visitor.visit_view_item(view_item, env.clone())
            }
            foreach foreign_item in foreign_module.items.iter() {
                visitor.visit_foreign_item(*foreign_item, env.clone())
            }
        }
        item_ty(ref typ, ref type_parameters) => {
            visitor.visit_ty(typ, env.clone());
            visitor.visit_generics(type_parameters, env)
        }
        item_enum(ref enum_definition, ref type_parameters) => {
            visitor.visit_generics(type_parameters, env.clone());
            visit_enum_def(visitor, enum_definition, type_parameters, env)
        }
        item_impl(ref type_parameters,
                  ref trait_references,
                  ref typ,
                  ref methods) => {
            visitor.visit_generics(type_parameters, env.clone());
            foreach trait_reference in trait_references.iter() {
                visit_trait_ref(visitor, trait_reference, env.clone())
            }
            visitor.visit_ty(typ, env.clone());
            foreach method in methods.iter() {
                visit_method_helper(visitor, *method, env.clone())
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
            foreach trait_path in trait_paths.iter() {
                visit_path(visitor, &trait_path.path, env.clone())
            }
            foreach method in methods.iter() {
                visitor.visit_trait_method(method, env.clone())
            }
        }
        item_mac(ref macro) => visit_mac(visitor, macro, env),
    }
}

pub fn visit_enum_def<E:Clone>(visitor: @Visitor<E>,
                               enum_definition: &ast::enum_def,
                               generics: &Generics,
                               env: E) {
    foreach variant in enum_definition.variants.iter() {
        match variant.node.kind {
            tuple_variant_kind(ref variant_arguments) => {
                foreach variant_argument in variant_arguments.iter() {
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

pub fn skip_ty<E>(_: @Visitor<E>, _: &Ty, _: E) {
    // Empty!
}

pub fn visit_ty<E:Clone>(visitor: @Visitor<E>, typ: &Ty, env: E) {
    match typ.node {
        ty_box(ref mutable_type) | ty_uniq(ref mutable_type) |
        ty_vec(ref mutable_type) | ty_ptr(ref mutable_type) |
        ty_rptr(_, ref mutable_type) => {
            visitor.visit_ty(mutable_type.ty, env)
        }
        ty_tup(ref tuple_element_types) => {
            foreach tuple_element_type in tuple_element_types.iter() {
                visitor.visit_ty(tuple_element_type, env.clone())
            }
        }
        ty_closure(ref function_declaration) => {
             foreach argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(&argument.ty, env.clone())
             }
             visitor.visit_ty(&function_declaration.decl.output, env.clone());
             foreach bounds in function_declaration.bounds.iter() {
                visit_ty_param_bounds(visitor, bounds, env.clone())
             }
        }
        ty_bare_fn(ref function_declaration) => {
            foreach argument in function_declaration.decl.inputs.iter() {
                visitor.visit_ty(&argument.ty, env.clone())
            }
            visitor.visit_ty(&function_declaration.decl.output, env.clone())
        }
        ty_path(ref path, ref bounds, _) => {
            visit_path(visitor, path, env.clone());
            foreach bounds in bounds.iter() {
                visit_ty_param_bounds(visitor, bounds, env.clone())
            }
        }
        ty_fixed_length_vec(ref mutable_type, expression) => {
            visitor.visit_ty(mutable_type.ty, env.clone());
            visitor.visit_expr(expression, env)
        }
        ty_nil | ty_bot | ty_mac(_) | ty_infer => ()
    }
}

pub fn visit_path<E:Clone>(visitor: @Visitor<E>, path: &Path, env: E) {
    foreach typ in path.types.iter() {
        visitor.visit_ty(typ, env.clone())
    }
}

pub fn visit_pat<E:Clone>(visitor: @Visitor<E>, pattern: &pat, env: E) {
    match pattern.node {
        pat_enum(ref path, ref children) => {
            visit_path(visitor, path, env.clone());
            foreach children in children.iter() {
                foreach child in children.iter() {
                    visitor.visit_pat(*child, env.clone())
                }
            }
        }
        pat_struct(ref path, ref fields, _) => {
            visit_path(visitor, path, env.clone());
            foreach field in fields.iter() {
                visitor.visit_pat(field.pat, env.clone())
            }
        }
        pat_tup(ref tuple_elements) => {
            foreach tuple_element in tuple_elements.iter() {
                visitor.visit_pat(*tuple_element, env.clone())
            }
        }
        pat_box(subpattern) |
        pat_uniq(subpattern) |
        pat_region(subpattern) => {
            visitor.visit_pat(subpattern, env)
        }
        pat_ident(_, ref path, ref optional_subpattern) => {
            visit_path(visitor, path, env.clone());
            match *optional_subpattern {
                None => {}
                Some(subpattern) => visitor.visit_pat(subpattern, env),
            }
        }
        pat_lit(expression) => visitor.visit_expr(expression, env),
        pat_range(lower_bound, upper_bound) => {
            visitor.visit_expr(lower_bound, env.clone());
            visitor.visit_expr(upper_bound, env)
        }
        pat_wild => (),
        pat_vec(ref prepattern, ref slice_pattern, ref postpatterns) => {
            foreach prepattern in prepattern.iter() {
                visitor.visit_pat(*prepattern, env.clone())
            }
            foreach slice_pattern in slice_pattern.iter() {
                visitor.visit_pat(*slice_pattern, env.clone())
            }
            foreach postpattern in postpatterns.iter() {
                visitor.visit_pat(*postpattern, env.clone())
            }
        }
    }
}

pub fn visit_foreign_item<E:Clone>(visitor: @Visitor<E>,
                                   foreign_item: &foreign_item,
                                   env: E) {
    match foreign_item.node {
        foreign_item_fn(ref function_declaration, ref generics) => {
            visit_fn_decl(visitor, function_declaration, env.clone());
            visitor.visit_generics(generics, env)
        }
        foreign_item_static(ref typ, _) => visitor.visit_ty(typ, env),
    }
}

pub fn visit_ty_param_bounds<E:Clone>(visitor: @Visitor<E>,
                                      bounds: &OptVec<TyParamBound>,
                                      env: E) {
    foreach bound in bounds.iter() {
        match *bound {
            TraitTyParamBound(ref typ) => {
                visit_trait_ref(visitor, typ, env.clone())
            }
            RegionTyParamBound => {}
        }
    }
}

pub fn visit_generics<E:Clone>(visitor: @Visitor<E>,
                               generics: &Generics,
                               env: E) {
    foreach type_parameter in generics.ty_params.iter() {
        visit_ty_param_bounds(visitor, &type_parameter.bounds, env.clone())
    }
}

pub fn visit_fn_decl<E:Clone>(visitor: @Visitor<E>,
                              function_declaration: &fn_decl,
                              env: E) {
    foreach argument in function_declaration.inputs.iter() {
        visitor.visit_pat(argument.pat, env.clone());
        visitor.visit_ty(&argument.ty, env.clone())
    }
    visitor.visit_ty(&function_declaration.output, env)
}

// Note: there is no visit_method() method in the visitor, instead override
// visit_fn() and check for fk_method().  I named this visit_method_helper()
// because it is not a default impl of any method, though I doubt that really
// clarifies anything. - Niko
pub fn visit_method_helper<E:Clone>(visitor: @Visitor<E>,
                                    method: &method,
                                    env: E) {
    visitor.visit_fn(&fk_method(method.ident, &method.generics, method),
                     &method.decl,
                     &method.body,
                     method.span,
                     method.id,
                     env)
}

pub fn visit_fn<E:Clone>(visitor: @Visitor<E>,
                         function_kind: &fn_kind,
                         function_declaration: &fn_decl,
                         function_body: &Block,
                         _: span,
                         _: NodeId,
                         env: E) {
    visit_fn_decl(visitor, function_declaration, env.clone());
    let generics = generics_of_fn(function_kind);
    visitor.visit_generics(&generics, env.clone());
    visitor.visit_block(function_body, env)
}

pub fn visit_ty_method<E:Clone>(visitor: @Visitor<E>,
                                method_type: &TypeMethod,
                                env: E) {
    foreach argument_type in method_type.decl.inputs.iter() {
        visitor.visit_ty(&argument_type.ty, env.clone())
    }
    visitor.visit_generics(&method_type.generics, env.clone());
    visitor.visit_ty(&method_type.decl.output, env.clone())
}

pub fn visit_trait_method<E:Clone>(visitor: @Visitor<E>,
                                   trait_method: &trait_method,
                                   env: E) {
    match *trait_method {
        required(ref method_type) => {
            visitor.visit_ty_method(method_type, env)
        }
        provided(method) => visit_method_helper(visitor, method, env),
    }
}

pub fn visit_struct_def<E:Clone>(visitor: @Visitor<E>,
                                 struct_definition: @struct_def,
                                 _: ast::ident,
                                 _: &Generics,
                                 _: NodeId,
                                 env: E) {
    foreach field in struct_definition.fields.iter() {
        visitor.visit_struct_field(*field, env.clone())
    }
}

pub fn visit_struct_field<E:Clone>(visitor: @Visitor<E>,
                                   struct_field: &struct_field,
                                   env: E) {
    visitor.visit_ty(&struct_field.node.ty, env)
}

pub fn visit_block<E:Clone>(visitor: @Visitor<E>, block: &Block, env: E) {
    foreach view_item in block.view_items.iter() {
        visitor.visit_view_item(view_item, env.clone())
    }
    foreach statement in block.stmts.iter() {
        visitor.visit_stmt(*statement, env.clone())
    }
    visit_expr_opt(visitor, block.expr, env)
}

pub fn visit_stmt<E>(visitor: @Visitor<E>, statement: &stmt, env: E) {
    match statement.node {
        stmt_decl(declaration, _) => visitor.visit_decl(declaration, env),
        stmt_expr(expression, _) | stmt_semi(expression, _) => {
            visitor.visit_expr(expression, env)
        }
        stmt_mac(ref macro, _) => visit_mac(visitor, macro, env),
    }
}

pub fn visit_decl<E:Clone>(visitor: @Visitor<E>, declaration: &decl, env: E) {
    match declaration.node {
        decl_local(ref local) => visitor.visit_local(*local, env),
        decl_item(item) => visitor.visit_item(item, env),
    }
}

pub fn visit_expr_opt<E>(visitor: @Visitor<E>,
                         optional_expression: Option<@expr>,
                         env: E) {
    match optional_expression {
        None => {}
        Some(expression) => visitor.visit_expr(expression, env),
    }
}

pub fn visit_exprs<E:Clone>(visitor: @Visitor<E>,
                            expressions: &[@expr],
                            env: E) {
    foreach expression in expressions.iter() {
        visitor.visit_expr(*expression, env.clone())
    }
}

pub fn visit_mac<E>(_: @Visitor<E>, _: &mac, _: E) {
    // Empty!
}

pub fn visit_expr<E:Clone>(visitor: @Visitor<E>, expression: @expr, env: E) {
    match expression.node {
        expr_vstore(subexpression, _) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        expr_vec(ref subexpressions, _) => {
            visit_exprs(visitor, *subexpressions, env.clone())
        }
        expr_repeat(element, count, _) => {
            visitor.visit_expr(element, env.clone());
            visitor.visit_expr(count, env.clone())
        }
        expr_struct(ref path, ref fields, optional_base) => {
            visit_path(visitor, path, env.clone());
            foreach field in fields.iter() {
                visitor.visit_expr(field.expr, env.clone())
            }
            visit_expr_opt(visitor, optional_base, env.clone())
        }
        expr_tup(ref subexpressions) => {
            foreach subexpression in subexpressions.iter() {
                visitor.visit_expr(*subexpression, env.clone())
            }
        }
        expr_call(callee_expression, ref arguments, _) => {
            foreach argument in arguments.iter() {
                visitor.visit_expr(*argument, env.clone())
            }
            visitor.visit_expr(callee_expression, env.clone())
        }
        expr_method_call(_, callee, _, ref types, ref arguments, _) => {
            visit_exprs(visitor, *arguments, env.clone());
            foreach typ in types.iter() {
                visitor.visit_ty(typ, env.clone())
            }
            visitor.visit_expr(callee, env.clone())
        }
        expr_binary(_, _, left_expression, right_expression) => {
            visitor.visit_expr(left_expression, env.clone());
            visitor.visit_expr(right_expression, env.clone())
        }
        expr_addr_of(_, subexpression) |
        expr_unary(_, _, subexpression) |
        expr_loop_body(subexpression) |
        expr_do_body(subexpression) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        expr_lit(_) => {}
        expr_cast(subexpression, ref typ) => {
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_ty(typ, env.clone())
        }
        expr_if(head_expression, ref if_block, optional_else) => {
            visitor.visit_expr(head_expression, env.clone());
            visitor.visit_block(if_block, env.clone());
            visit_expr_opt(visitor, optional_else, env.clone())
        }
        expr_while(subexpression, ref block) => {
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_block(block, env.clone())
        }
        expr_for_loop(pattern, subexpression, ref block) => {
            visitor.visit_pat(pattern, env.clone());
            visitor.visit_expr(subexpression, env.clone());
            visitor.visit_block(block, env.clone())
        }
        expr_loop(ref block, _) => visitor.visit_block(block, env.clone()),
        expr_match(subexpression, ref arms) => {
            visitor.visit_expr(subexpression, env.clone());
            foreach arm in arms.iter() {
                visitor.visit_arm(arm, env.clone())
            }
        }
        expr_fn_block(ref function_declaration, ref body) => {
            visitor.visit_fn(&fk_fn_block,
                             function_declaration,
                             body,
                             expression.span,
                             expression.id,
                             env.clone())
        }
        expr_block(ref block) => visitor.visit_block(block, env.clone()),
        expr_assign(left_hand_expression, right_hand_expression) => {
            visitor.visit_expr(right_hand_expression, env.clone());
            visitor.visit_expr(left_hand_expression, env.clone())
        }
        expr_assign_op(_, _, left_expression, right_expression) => {
            visitor.visit_expr(right_expression, env.clone());
            visitor.visit_expr(left_expression, env.clone())
        }
        expr_field(subexpression, _, ref types) => {
            visitor.visit_expr(subexpression, env.clone());
            foreach typ in types.iter() {
                visitor.visit_ty(typ, env.clone())
            }
        }
        expr_index(_, main_expression, index_expression) => {
            visitor.visit_expr(main_expression, env.clone());
            visitor.visit_expr(index_expression, env.clone())
        }
        expr_path(ref path) => visit_path(visitor, path, env.clone()),
        expr_self | expr_break(_) | expr_again(_) => {}
        expr_ret(optional_expression) => {
            visit_expr_opt(visitor, optional_expression, env.clone())
        }
        expr_log(level, subexpression) => {
            visitor.visit_expr(level, env.clone());
            visitor.visit_expr(subexpression, env.clone());
        }
        expr_mac(ref macro) => visit_mac(visitor, macro, env.clone()),
        expr_paren(subexpression) => {
            visitor.visit_expr(subexpression, env.clone())
        }
        expr_inline_asm(ref assembler) => {
            foreach &(_, input) in assembler.inputs.iter() {
                visitor.visit_expr(input, env.clone())
            }
            foreach &(_, output) in assembler.outputs.iter() {
                visitor.visit_expr(output, env.clone())
            }
        }
    }

    visitor.visit_expr_post(expression, env.clone())
}

pub fn visit_arm<E:Clone>(visitor: @Visitor<E>, arm: &arm, env: E) {
    foreach pattern in arm.pats.iter() {
        visitor.visit_pat(*pattern, env.clone())
    }
    visit_expr_opt(visitor, arm.guard, env.clone());
    visitor.visit_block(&arm.body, env)
}

// Simpler, non-context passing interface. Always walks the whole tree, simply
// calls the given functions on the nodes.

pub trait SimpleVisitor {
    fn visit_mod(@mut self, &_mod, span, NodeId);
    fn visit_view_item(@mut self, &view_item);
    fn visit_foreign_item(@mut self, @foreign_item);
    fn visit_item(@mut self, @item);
    fn visit_local(@mut self, @Local);
    fn visit_block(@mut self, &Block);
    fn visit_stmt(@mut self, @stmt);
    fn visit_arm(@mut self, &arm);
    fn visit_pat(@mut self, @pat);
    fn visit_decl(@mut self, @decl);
    fn visit_expr(@mut self, @expr);
    fn visit_expr_post(@mut self, @expr);
    fn visit_ty(@mut self, &Ty);
    fn visit_generics(@mut self, &Generics);
    fn visit_fn(@mut self, &fn_kind, &fn_decl, &Block, span, NodeId);
    fn visit_ty_method(@mut self, &TypeMethod);
    fn visit_trait_method(@mut self, &trait_method);
    fn visit_struct_def(@mut self, @struct_def, ident, &Generics, NodeId);
    fn visit_struct_field(@mut self, @struct_field);
    fn visit_struct_method(@mut self, @method);
}

pub struct SimpleVisitorVisitor {
    simple_visitor: @SimpleVisitor,
}

impl Visitor<()> for SimpleVisitorVisitor {
    fn visit_mod(@mut self,
                 module: &_mod,
                 span: span,
                 node_id: NodeId,
                 env: ()) {
        self.simple_visitor.visit_mod(module, span, node_id);
        visit_mod(self as @Visitor<()>, module, env)
    }
    fn visit_view_item(@mut self, view_item: &view_item, env: ()) {
        self.simple_visitor.visit_view_item(view_item);
        visit_view_item(self as @Visitor<()>, view_item, env)
    }
    fn visit_foreign_item(@mut self, foreign_item: @foreign_item, env: ()) {
        self.simple_visitor.visit_foreign_item(foreign_item);
        visit_foreign_item(self as @Visitor<()>, foreign_item, env)
    }
    fn visit_item(@mut self, item: @item, env: ()) {
        self.simple_visitor.visit_item(item);
        visit_item(self as @Visitor<()>, item, env)
    }
    fn visit_local(@mut self, local: @Local, env: ()) {
        self.simple_visitor.visit_local(local);
        visit_local(self as @Visitor<()>, local, env)
    }
    fn visit_block(@mut self, block: &Block, env: ()) {
        self.simple_visitor.visit_block(block);
        visit_block(self as @Visitor<()>, block, env)
    }
    fn visit_stmt(@mut self, statement: @stmt, env: ()) {
        self.simple_visitor.visit_stmt(statement);
        visit_stmt(self as @Visitor<()>, statement, env)
    }
    fn visit_arm(@mut self, arm: &arm, env: ()) {
        self.simple_visitor.visit_arm(arm);
        visit_arm(self as @Visitor<()>, arm, env)
    }
    fn visit_pat(@mut self, pattern: @pat, env: ()) {
        self.simple_visitor.visit_pat(pattern);
        visit_pat(self as @Visitor<()>, pattern, env)
    }
    fn visit_decl(@mut self, declaration: @decl, env: ()) {
        self.simple_visitor.visit_decl(declaration);
        visit_decl(self as @Visitor<()>, declaration, env)
    }
    fn visit_expr(@mut self, expression: @expr, env: ()) {
        self.simple_visitor.visit_expr(expression);
        visit_expr(self as @Visitor<()>, expression, env)
    }
    fn visit_expr_post(@mut self, expression: @expr, _: ()) {
        self.simple_visitor.visit_expr_post(expression)
    }
    fn visit_ty(@mut self, typ: &Ty, env: ()) {
        self.simple_visitor.visit_ty(typ);
        visit_ty(self as @Visitor<()>, typ, env)
    }
    fn visit_generics(@mut self, generics: &Generics, env: ()) {
        self.simple_visitor.visit_generics(generics);
        visit_generics(self as @Visitor<()>, generics, env)
    }
    fn visit_fn(@mut self,
                function_kind: &fn_kind,
                function_declaration: &fn_decl,
                block: &Block,
                span: span,
                node_id: NodeId,
                env: ()) {
        self.simple_visitor.visit_fn(function_kind,
                                     function_declaration,
                                     block,
                                     span,
                                     node_id);
        visit_fn(self as @Visitor<()>,
                 function_kind,
                 function_declaration,
                 block,
                 span,
                 node_id,
                 env)
    }
    fn visit_ty_method(@mut self, method_type: &TypeMethod, env: ()) {
        self.simple_visitor.visit_ty_method(method_type);
        visit_ty_method(self as @Visitor<()>, method_type, env)
    }
    fn visit_trait_method(@mut self, trait_method: &trait_method, env: ()) {
        self.simple_visitor.visit_trait_method(trait_method);
        visit_trait_method(self as @Visitor<()>, trait_method, env)
    }
    fn visit_struct_def(@mut self,
                        struct_definition: @struct_def,
                        identifier: ident,
                        generics: &Generics,
                        node_id: NodeId,
                        env: ()) {
        self.simple_visitor.visit_struct_def(struct_definition,
                                             identifier,
                                             generics,
                                             node_id);
        visit_struct_def(self as @Visitor<()>,
                         struct_definition,
                         identifier,
                         generics,
                         node_id,
                         env)
    }
    fn visit_struct_field(@mut self, struct_field: @struct_field, env: ()) {
        self.simple_visitor.visit_struct_field(struct_field);
        visit_struct_field(self as @Visitor<()>, struct_field, env)
    }
}

