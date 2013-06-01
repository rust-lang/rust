// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Enforces the Rust effect system. Currently there is just one effect,
/// `unsafe`.

use middle::ty::{ty_bare_fn, ty_closure, ty_ptr};
use middle::ty;
use middle::typeck::method_map;
use util::ppaux;

use syntax::ast::{deref, expr_call, expr_inline_asm, expr_method_call};
use syntax::ast::{expr_unary, node_id, unsafe_blk, unsafe_fn};
use syntax::ast;
use syntax::codemap::span;
use syntax::visit::{fk_item_fn, fk_method};
use syntax::visit;

#[deriving(Eq)]
enum UnsafeContext {
    SafeContext,
    UnsafeFn,
    UnsafeBlock(node_id),
}

struct Context {
    /// The method map.
    method_map: method_map,
    /// Whether we're in an unsafe context.
    unsafe_context: UnsafeContext,
}

fn type_is_unsafe_function(ty: ty::t) -> bool {
    match ty::get(ty).sty {
        ty_bare_fn(ref f) => f.purity == unsafe_fn,
        ty_closure(ref f) => f.purity == unsafe_fn,
        _ => false,
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: method_map,
                   crate: @ast::crate) {
    let context = @mut Context {
        method_map: method_map,
        unsafe_context: SafeContext,
    };

    let require_unsafe: @fn(span: span,
                            description: &str) = |span, description| {
        match context.unsafe_context {
            SafeContext => {
                // Report an error.
                tcx.sess.span_err(span,
                                  fmt!("%s requires unsafe function or block",
                                       description))
            }
            UnsafeBlock(block_id) => {
                // OK, but record this.
                debug!("effect: recording unsafe block as used: %?", block_id);
                let _ = tcx.used_unsafe.insert(block_id);
            }
            UnsafeFn => {}
        }
    };

    let visitor = visit::mk_vt(@visit::Visitor {
        visit_fn: |fn_kind, fn_decl, block, span, node_id, _, visitor| {
            let (is_item_fn, is_unsafe_fn) = match *fn_kind {
                fk_item_fn(_, _, purity, _) => (true, purity == unsafe_fn),
                fk_method(_, _, method) => (true, method.purity == unsafe_fn),
                _ => (false, false),
            };

            let old_unsafe_context = context.unsafe_context;
            if is_unsafe_fn {
                context.unsafe_context = UnsafeFn
            } else if is_item_fn {
                context.unsafe_context = SafeContext
            }

            visit::visit_fn(fn_kind,
                            fn_decl,
                            block,
                            span,
                            node_id,
                            (),
                            visitor);

            context.unsafe_context = old_unsafe_context
        },

        visit_block: |block, _, visitor| {
            let old_unsafe_context = context.unsafe_context;
            if block.node.rules == unsafe_blk &&
                    context.unsafe_context == SafeContext {
                context.unsafe_context = UnsafeBlock(block.node.id)
            }

            visit::visit_block(block, (), visitor);

            context.unsafe_context = old_unsafe_context
        },

        visit_expr: |expr, _, visitor| {
            match expr.node {
                expr_method_call(callee_id, _, _, _, _, _) => {
                    let base_type = ty::node_id_to_type(tcx, callee_id);
                    debug!("effect: method call case, base type is %s",
                           ppaux::ty_to_str(tcx, base_type));
                    if type_is_unsafe_function(base_type) {
                        require_unsafe(expr.span,
                                       "invocation of unsafe method")
                    }
                }
                expr_call(base, _, _) => {
                    let base_type = ty::node_id_to_type(tcx, base.id);
                    debug!("effect: call case, base type is %s",
                           ppaux::ty_to_str(tcx, base_type));
                    if type_is_unsafe_function(base_type) {
                        require_unsafe(expr.span, "call to unsafe function")
                    }
                }
                expr_unary(_, deref, base) => {
                    let base_type = ty::node_id_to_type(tcx, base.id);
                    debug!("effect: unary case, base type is %s",
                           ppaux::ty_to_str(tcx, base_type));
                    match ty::get(base_type).sty {
                        ty_ptr(_) => {
                            require_unsafe(expr.span,
                                           "dereference of unsafe pointer")
                        }
                        _ => {}
                    }
                }
                expr_inline_asm(*) => {
                    require_unsafe(expr.span, "use of inline assembly")
                }
                _ => {}
            }

            visit::visit_expr(expr, (), visitor)
        },

        .. *visit::default_visitor()
    });

    visit::visit_crate(crate, (), visitor)
}

