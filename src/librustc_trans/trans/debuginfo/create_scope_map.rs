// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::metadata::file_metadata;
use super::utils::DIB;

use llvm;
use llvm::debuginfo::{DIScope, DISubprogram};
use trans::common::CrateContext;
use middle::pat_util;
use rustc::util::nodemap::NodeMap;

use libc::c_uint;
use syntax::codemap::{Span, Pos};
use syntax::{ast, codemap};

use rustc_front;
use rustc_front::hir;

// This procedure builds the *scope map* for a given function, which maps any
// given ast::NodeId in the function's AST to the correct DIScope metadata instance.
//
// This builder procedure walks the AST in execution order and keeps track of
// what belongs to which scope, creating DIScope DIEs along the way, and
// introducing *artificial* lexical scope descriptors where necessary. These
// artificial scopes allow GDB to correctly handle name shadowing.
pub fn create_scope_map(cx: &CrateContext,
                        args: &[hir::Arg],
                        fn_entry_block: &hir::Block,
                        fn_metadata: DISubprogram,
                        fn_ast_id: ast::NodeId)
                        -> NodeMap<DIScope> {
    let mut scope_map = NodeMap();

    let def_map = &cx.tcx().def_map;

    let mut scope_stack = vec!(ScopeStackEntry { scope_metadata: fn_metadata, name: None });
    scope_map.insert(fn_ast_id, fn_metadata);

    // Push argument identifiers onto the stack so arguments integrate nicely
    // with variable shadowing.
    for arg in args {
        pat_util::pat_bindings(def_map, &*arg.pat, |_, node_id, _, path1| {
            scope_stack.push(ScopeStackEntry { scope_metadata: fn_metadata,
                                               name: Some(path1.node) });
            scope_map.insert(node_id, fn_metadata);
        })
    }

    // Clang creates a separate scope for function bodies, so let's do this too.
    with_new_scope(cx,
                   fn_entry_block.span,
                   &mut scope_stack,
                   &mut scope_map,
                   |cx, scope_stack, scope_map| {
        walk_block(cx, fn_entry_block, scope_stack, scope_map);
    });

    return scope_map;
}

// local helper functions for walking the AST.
fn with_new_scope<F>(cx: &CrateContext,
                     scope_span: Span,
                     scope_stack: &mut Vec<ScopeStackEntry> ,
                     scope_map: &mut NodeMap<DIScope>,
                     inner_walk: F) where
    F: FnOnce(&CrateContext, &mut Vec<ScopeStackEntry>, &mut NodeMap<DIScope>),
{
    // Create a new lexical scope and push it onto the stack
    let loc = cx.sess().codemap().lookup_char_pos(scope_span.lo);
    let file_metadata = file_metadata(cx, &loc.file.name);
    let parent_scope = scope_stack.last().unwrap().scope_metadata;

    let scope_metadata = unsafe {
        llvm::LLVMDIBuilderCreateLexicalBlock(
            DIB(cx),
            parent_scope,
            file_metadata,
            loc.line as c_uint,
            loc.col.to_usize() as c_uint)
    };

    scope_stack.push(ScopeStackEntry { scope_metadata: scope_metadata, name: None });

    inner_walk(cx, scope_stack, scope_map);

    // pop artificial scopes
    while scope_stack.last().unwrap().name.is_some() {
        scope_stack.pop();
    }

    if scope_stack.last().unwrap().scope_metadata != scope_metadata {
        cx.sess().span_bug(scope_span, "debuginfo: Inconsistency in scope management.");
    }

    scope_stack.pop();
}

struct ScopeStackEntry {
    scope_metadata: DIScope,
    name: Option<ast::Name>
}

fn walk_block(cx: &CrateContext,
              block: &hir::Block,
              scope_stack: &mut Vec<ScopeStackEntry> ,
              scope_map: &mut NodeMap<DIScope>) {
    scope_map.insert(block.id, scope_stack.last().unwrap().scope_metadata);

    // The interesting things here are statements and the concluding expression.
    for statement in &block.stmts {
        scope_map.insert(rustc_front::util::stmt_id(&**statement),
                         scope_stack.last().unwrap().scope_metadata);

        match statement.node {
            hir::StmtDecl(ref decl, _) =>
                walk_decl(cx, &**decl, scope_stack, scope_map),
            hir::StmtExpr(ref exp, _) |
            hir::StmtSemi(ref exp, _) =>
                walk_expr(cx, &**exp, scope_stack, scope_map),
        }
    }

    if let Some(ref exp) = block.expr {
        walk_expr(cx, &**exp, scope_stack, scope_map);
    }
}

fn walk_decl(cx: &CrateContext,
             decl: &hir::Decl,
             scope_stack: &mut Vec<ScopeStackEntry> ,
             scope_map: &mut NodeMap<DIScope>) {
    match *decl {
        codemap::Spanned { node: hir::DeclLocal(ref local), .. } => {
            scope_map.insert(local.id, scope_stack.last().unwrap().scope_metadata);

            walk_pattern(cx, &*local.pat, scope_stack, scope_map);

            if let Some(ref exp) = local.init {
                walk_expr(cx, &**exp, scope_stack, scope_map);
            }
        }
        _ => ()
    }
}

fn walk_pattern(cx: &CrateContext,
                pat: &hir::Pat,
                scope_stack: &mut Vec<ScopeStackEntry> ,
                scope_map: &mut NodeMap<DIScope>) {

    let def_map = &cx.tcx().def_map;

    // Unfortunately, we cannot just use pat_util::pat_bindings() or
    // ast_util::walk_pat() here because we have to visit *all* nodes in
    // order to put them into the scope map. The above functions don't do that.
    match pat.node {
        hir::PatIdent(_, ref path1, ref sub_pat_opt) => {

            // Check if this is a binding. If so we need to put it on the
            // scope stack and maybe introduce an artificial scope
            if pat_util::pat_is_binding(def_map, &*pat) {

                let name = path1.node.name;

                // LLVM does not properly generate 'DW_AT_start_scope' fields
                // for variable DIEs. For this reason we have to introduce
                // an artificial scope at bindings whenever a variable with
                // the same name is declared in *any* parent scope.
                //
                // Otherwise the following error occurs:
                //
                // let x = 10;
                //
                // do_something(); // 'gdb print x' correctly prints 10
                //
                // {
                //     do_something(); // 'gdb print x' prints 0, because it
                //                     // already reads the uninitialized 'x'
                //                     // from the next line...
                //     let x = 100;
                //     do_something(); // 'gdb print x' correctly prints 100
                // }

                // Is there already a binding with that name?
                // N.B.: this comparison must be UNhygienic... because
                // gdb knows nothing about the context, so any two
                // variables with the same name will cause the problem.
                let need_new_scope = scope_stack
                    .iter()
                    .any(|entry| entry.name == Some(name));

                if need_new_scope {
                    // Create a new lexical scope and push it onto the stack
                    let loc = cx.sess().codemap().lookup_char_pos(pat.span.lo);
                    let file_metadata = file_metadata(cx, &loc.file.name);
                    let parent_scope = scope_stack.last().unwrap().scope_metadata;

                    let scope_metadata = unsafe {
                        llvm::LLVMDIBuilderCreateLexicalBlock(
                            DIB(cx),
                            parent_scope,
                            file_metadata,
                            loc.line as c_uint,
                            loc.col.to_usize() as c_uint)
                    };

                    scope_stack.push(ScopeStackEntry {
                        scope_metadata: scope_metadata,
                        name: Some(name)
                    });

                } else {
                    // Push a new entry anyway so the name can be found
                    let prev_metadata = scope_stack.last().unwrap().scope_metadata;
                    scope_stack.push(ScopeStackEntry {
                        scope_metadata: prev_metadata,
                        name: Some(name)
                    });
                }
            }

            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);

            if let Some(ref sub_pat) = *sub_pat_opt {
                walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
            }
        }

        hir::PatWild => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);
        }

        hir::PatEnum(_, ref sub_pats_opt) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);

            if let Some(ref sub_pats) = *sub_pats_opt {
                for p in sub_pats {
                    walk_pattern(cx, &**p, scope_stack, scope_map);
                }
            }
        }

        hir::PatQPath(..) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);
        }

        hir::PatStruct(_, ref field_pats, _) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);

            for &codemap::Spanned {
                node: hir::FieldPat { pat: ref sub_pat, .. },
                ..
            } in field_pats {
                walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
            }
        }

        hir::PatTup(ref sub_pats) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);

            for sub_pat in sub_pats {
                walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
            }
        }

        hir::PatBox(ref sub_pat) | hir::PatRegion(ref sub_pat, _) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);
            walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
        }

        hir::PatLit(ref exp) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);
            walk_expr(cx, &**exp, scope_stack, scope_map);
        }

        hir::PatRange(ref exp1, ref exp2) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);
            walk_expr(cx, &**exp1, scope_stack, scope_map);
            walk_expr(cx, &**exp2, scope_stack, scope_map);
        }

        hir::PatVec(ref front_sub_pats, ref middle_sub_pats, ref back_sub_pats) => {
            scope_map.insert(pat.id, scope_stack.last().unwrap().scope_metadata);

            for sub_pat in front_sub_pats {
                walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
            }

            if let Some(ref sub_pat) = *middle_sub_pats {
                walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
            }

            for sub_pat in back_sub_pats {
                walk_pattern(cx, &**sub_pat, scope_stack, scope_map);
            }
        }
    }
}

fn walk_expr(cx: &CrateContext,
             exp: &hir::Expr,
             scope_stack: &mut Vec<ScopeStackEntry> ,
             scope_map: &mut NodeMap<DIScope>) {

    scope_map.insert(exp.id, scope_stack.last().unwrap().scope_metadata);

    match exp.node {
        hir::ExprLit(_)   |
        hir::ExprBreak(_) |
        hir::ExprAgain(_) |
        hir::ExprPath(..) => {}

        hir::ExprCast(ref sub_exp, _)     |
        hir::ExprAddrOf(_, ref sub_exp)  |
        hir::ExprField(ref sub_exp, _) |
        hir::ExprTupField(ref sub_exp, _) =>
            walk_expr(cx, &**sub_exp, scope_stack, scope_map),

        hir::ExprBox(ref sub_expr) => {
            walk_expr(cx, &**sub_expr, scope_stack, scope_map);
        }

        hir::ExprRet(ref exp_opt) => match *exp_opt {
            Some(ref sub_exp) => walk_expr(cx, &**sub_exp, scope_stack, scope_map),
            None => ()
        },

        hir::ExprUnary(_, ref sub_exp) => {
            walk_expr(cx, &**sub_exp, scope_stack, scope_map);
        }

        hir::ExprAssignOp(_, ref lhs, ref rhs) |
        hir::ExprIndex(ref lhs, ref rhs) |
        hir::ExprBinary(_, ref lhs, ref rhs)    => {
            walk_expr(cx, &**lhs, scope_stack, scope_map);
            walk_expr(cx, &**rhs, scope_stack, scope_map);
        }

        hir::ExprRange(ref start, ref end) => {
            start.as_ref().map(|e| walk_expr(cx, &**e, scope_stack, scope_map));
            end.as_ref().map(|e| walk_expr(cx, &**e, scope_stack, scope_map));
        }

        hir::ExprVec(ref init_expressions) |
        hir::ExprTup(ref init_expressions) => {
            for ie in init_expressions {
                walk_expr(cx, &**ie, scope_stack, scope_map);
            }
        }

        hir::ExprAssign(ref sub_exp1, ref sub_exp2) |
        hir::ExprRepeat(ref sub_exp1, ref sub_exp2) => {
            walk_expr(cx, &**sub_exp1, scope_stack, scope_map);
            walk_expr(cx, &**sub_exp2, scope_stack, scope_map);
        }

        hir::ExprIf(ref cond_exp, ref then_block, ref opt_else_exp) => {
            walk_expr(cx, &**cond_exp, scope_stack, scope_map);

            with_new_scope(cx,
                           then_block.span,
                           scope_stack,
                           scope_map,
                           |cx, scope_stack, scope_map| {
                walk_block(cx, &**then_block, scope_stack, scope_map);
            });

            match *opt_else_exp {
                Some(ref else_exp) =>
                    walk_expr(cx, &**else_exp, scope_stack, scope_map),
                _ => ()
            }
        }

        hir::ExprWhile(ref cond_exp, ref loop_body, _) => {
            walk_expr(cx, &**cond_exp, scope_stack, scope_map);

            with_new_scope(cx,
                           loop_body.span,
                           scope_stack,
                           scope_map,
                           |cx, scope_stack, scope_map| {
                walk_block(cx, &**loop_body, scope_stack, scope_map);
            })
        }

        hir::ExprLoop(ref block, _) |
        hir::ExprBlock(ref block)   => {
            with_new_scope(cx,
                           block.span,
                           scope_stack,
                           scope_map,
                           |cx, scope_stack, scope_map| {
                walk_block(cx, &**block, scope_stack, scope_map);
            })
        }

        hir::ExprClosure(_, ref decl, ref block) => {
            with_new_scope(cx,
                           block.span,
                           scope_stack,
                           scope_map,
                           |cx, scope_stack, scope_map| {
                for &hir::Arg { pat: ref pattern, .. } in &decl.inputs {
                    walk_pattern(cx, &**pattern, scope_stack, scope_map);
                }

                walk_block(cx, &**block, scope_stack, scope_map);
            })
        }

        hir::ExprCall(ref fn_exp, ref args) => {
            walk_expr(cx, &**fn_exp, scope_stack, scope_map);

            for arg_exp in args {
                walk_expr(cx, &**arg_exp, scope_stack, scope_map);
            }
        }

        hir::ExprMethodCall(_, _, ref args) => {
            for arg_exp in args {
                walk_expr(cx, &**arg_exp, scope_stack, scope_map);
            }
        }

        hir::ExprMatch(ref discriminant_exp, ref arms, _) => {
            walk_expr(cx, &**discriminant_exp, scope_stack, scope_map);

            // For each arm we have to first walk the pattern as these might
            // introduce new artificial scopes. It should be sufficient to
            // walk only one pattern per arm, as they all must contain the
            // same binding names.

            for arm_ref in arms {
                let arm_span = arm_ref.pats[0].span;

                with_new_scope(cx,
                               arm_span,
                               scope_stack,
                               scope_map,
                               |cx, scope_stack, scope_map| {
                    for pat in &arm_ref.pats {
                        walk_pattern(cx, &**pat, scope_stack, scope_map);
                    }

                    if let Some(ref guard_exp) = arm_ref.guard {
                        walk_expr(cx, &**guard_exp, scope_stack, scope_map)
                    }

                    walk_expr(cx, &*arm_ref.body, scope_stack, scope_map);
                })
            }
        }

        hir::ExprStruct(_, ref fields, ref base_exp) => {
            for &hir::Field { expr: ref exp, .. } in fields {
                walk_expr(cx, &**exp, scope_stack, scope_map);
            }

            match *base_exp {
                Some(ref exp) => walk_expr(cx, &**exp, scope_stack, scope_map),
                None => ()
            }
        }

        hir::ExprInlineAsm(hir::InlineAsm { ref inputs,
                                            ref outputs,
                                            .. }) => {
            // inputs, outputs: Vec<(String, P<Expr>)>
            for &(_, ref exp) in inputs {
                walk_expr(cx, &**exp, scope_stack, scope_map);
            }

            for &(_, ref exp, _) in outputs {
                walk_expr(cx, &**exp, scope_stack, scope_map);
            }
        }
    }
}
