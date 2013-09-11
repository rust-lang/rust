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

use syntax::ast::{UnDeref, ExprCall, ExprInlineAsm, ExprMethodCall};
use syntax::ast::{ExprUnary, unsafe_fn, ExprPath};
use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::{fk_item_fn, fk_method};
use syntax::visit;
use syntax::visit::{Visitor,fn_kind};
use syntax::ast::{fn_decl,Block,NodeId,Expr};

#[deriving(Eq)]
enum UnsafeContext {
    SafeContext,
    UnsafeFn,
    UnsafeBlock(ast::NodeId),
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

struct EffectCheckVisitor {
    tcx: ty::ctxt,
    context: @mut Context,
}

impl EffectCheckVisitor {
    fn require_unsafe(&mut self, span: Span, description: &str) {
        match self.context.unsafe_context {
            SafeContext => {
                // Report an error.
                self.tcx.sess.span_err(span,
                                  fmt!("%s requires unsafe function or block",
                                       description))
            }
            UnsafeBlock(block_id) => {
                // OK, but record this.
                debug!("effect: recording unsafe block as used: %?", block_id);
                let _ = self.tcx.used_unsafe.insert(block_id);
            }
            UnsafeFn => {}
        }
    }
}

impl Visitor<()> for EffectCheckVisitor {
    fn visit_fn(&mut self, fn_kind:&fn_kind, fn_decl:&fn_decl,
                block:&Block, span:Span, node_id:NodeId, _:()) {

            let (is_item_fn, is_unsafe_fn) = match *fn_kind {
                fk_item_fn(_, _, purity, _) => (true, purity == unsafe_fn),
                fk_method(_, _, method) => (true, method.purity == unsafe_fn),
                _ => (false, false),
            };

            let old_unsafe_context = self.context.unsafe_context;
            if is_unsafe_fn {
                self.context.unsafe_context = UnsafeFn
            } else if is_item_fn {
                self.context.unsafe_context = SafeContext
            }

            visit::walk_fn(self,
                           fn_kind,
                            fn_decl,
                            block,
                            span,
                            node_id,
                            ());

            self.context.unsafe_context = old_unsafe_context
    }

    fn visit_block(&mut self, block:&Block, _:()) {

            let old_unsafe_context = self.context.unsafe_context;
            let is_unsafe = match block.rules {
                ast::UnsafeBlock(*) => true, ast::DefaultBlock => false
            };
            if is_unsafe && self.context.unsafe_context == SafeContext {
                self.context.unsafe_context = UnsafeBlock(block.id)
            }

            visit::walk_block(self, block, ());

            self.context.unsafe_context = old_unsafe_context
    }

    fn visit_expr(&mut self, expr:@Expr, _:()) {

            match expr.node {
                ExprMethodCall(callee_id, _, _, _, _, _) => {
                    let base_type = ty::node_id_to_type(self.tcx, callee_id);
                    debug!("effect: method call case, base type is %s",
                           ppaux::ty_to_str(self.tcx, base_type));
                    if type_is_unsafe_function(base_type) {
                        self.require_unsafe(expr.span,
                                       "invocation of unsafe method")
                    }
                }
                ExprCall(base, _, _) => {
                    let base_type = ty::node_id_to_type(self.tcx, base.id);
                    debug!("effect: call case, base type is %s",
                           ppaux::ty_to_str(self.tcx, base_type));
                    if type_is_unsafe_function(base_type) {
                        self.require_unsafe(expr.span, "call to unsafe function")
                    }
                }
                ExprUnary(_, UnDeref, base) => {
                    let base_type = ty::node_id_to_type(self.tcx, base.id);
                    debug!("effect: unary case, base type is %s",
                           ppaux::ty_to_str(self.tcx, base_type));
                    match ty::get(base_type).sty {
                        ty_ptr(_) => {
                            self.require_unsafe(expr.span,
                                           "dereference of unsafe pointer")
                        }
                        _ => {}
                    }
                }
                ExprInlineAsm(*) => {
                    self.require_unsafe(expr.span, "use of inline assembly")
                }
                ExprPath(*) => {
                    match ty::resolve_expr(self.tcx, expr) {
                        ast::DefStatic(_, true) => {
                            self.require_unsafe(expr.span, "use of mutable static")
                        }
                        _ => {}
                    }
                }
                _ => {}
            }

            visit::walk_expr(self, expr, ());
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: method_map,
                   crate: &ast::Crate) {
    let context = @mut Context {
        method_map: method_map,
        unsafe_context: SafeContext,
    };

    let mut visitor = EffectCheckVisitor {
        tcx: tcx,
        context: context,
    };

    visit::walk_crate(&mut visitor, crate, ());
}
