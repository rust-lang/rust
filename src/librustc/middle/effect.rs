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

use middle::ty;
use middle::typeck::method_map;
use util::ppaux;

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;

#[deriving(Eq)]
enum UnsafeContext {
    SafeContext,
    UnsafeFn,
    UnsafeBlock(ast::NodeId),
}

fn type_is_unsafe_function(ty: ty::t) -> bool {
    match ty::get(ty).sty {
        ty::ty_bare_fn(ref f) => f.purity == ast::UnsafeFn,
        ty::ty_closure(ref f) => f.purity == ast::UnsafeFn,
        _ => false,
    }
}

struct EffectCheckVisitor {
    tcx: ty::ctxt,

    /// The method map.
    method_map: method_map,
    /// Whether we're in an unsafe context.
    unsafe_context: UnsafeContext,
}

impl EffectCheckVisitor {
    fn require_unsafe(&mut self, span: Span, description: &str) {
        match self.unsafe_context {
            SafeContext => {
                // Report an error.
                self.tcx.sess.span_err(span,
                                  format!("{} requires unsafe function or block",
                                       description))
            }
            UnsafeBlock(block_id) => {
                // OK, but record this.
                debug!("effect: recording unsafe block as used: {:?}", block_id);
                let mut used_unsafe = self.tcx.used_unsafe.borrow_mut();
                let _ = used_unsafe.get().insert(block_id);
            }
            UnsafeFn => {}
        }
    }

    fn check_str_index(&mut self, e: @ast::Expr) {
        let base_type = match e.node {
            ast::ExprIndex(_, base, _) => ty::node_id_to_type(self.tcx, base.id),
            _ => return
        };
        debug!("effect: checking index with base type {}",
                ppaux::ty_to_str(self.tcx, base_type));
        match ty::get(base_type).sty {
            ty::ty_estr(..) => {
                self.tcx.sess.span_err(e.span,
                    "modification of string types is not allowed");
            }
            _ => {}
        }
    }
}

impl Visitor<()> for EffectCheckVisitor {
    fn visit_fn(&mut self, fn_kind: &visit::FnKind, fn_decl: &ast::FnDecl,
                block: &ast::Block, span: Span, node_id: ast::NodeId, _:()) {

        let (is_item_fn, is_unsafe_fn) = match *fn_kind {
            visit::FkItemFn(_, _, purity, _) =>
                (true, purity == ast::UnsafeFn),
            visit::FkMethod(_, _, method) =>
                (true, method.purity == ast::UnsafeFn),
            _ => (false, false),
        };

        let old_unsafe_context = self.unsafe_context;
        if is_unsafe_fn {
            self.unsafe_context = UnsafeFn
        } else if is_item_fn {
            self.unsafe_context = SafeContext
        }

        visit::walk_fn(self, fn_kind, fn_decl, block, span, node_id, ());

        self.unsafe_context = old_unsafe_context
    }

    fn visit_block(&mut self, block: &ast::Block, _:()) {
        let old_unsafe_context = self.unsafe_context;
        let is_unsafe = match block.rules {
            ast::UnsafeBlock(..) => true, ast::DefaultBlock => false
        };
        if is_unsafe && self.unsafe_context == SafeContext {
            self.unsafe_context = UnsafeBlock(block.id)
        }

        visit::walk_block(self, block, ());

        self.unsafe_context = old_unsafe_context
    }

    fn visit_expr(&mut self, expr: &ast::Expr, _:()) {
        match expr.node {
            ast::ExprMethodCall(callee_id, _, _, _, _, _) => {
                let base_type = ty::node_id_to_type(self.tcx, callee_id);
                debug!("effect: method call case, base type is {}",
                       ppaux::ty_to_str(self.tcx, base_type));
                if type_is_unsafe_function(base_type) {
                    self.require_unsafe(expr.span,
                                        "invocation of unsafe method")
                }
            }
            ast::ExprCall(base, _, _) => {
                let base_type = ty::node_id_to_type(self.tcx, base.id);
                debug!("effect: call case, base type is {}",
                       ppaux::ty_to_str(self.tcx, base_type));
                if type_is_unsafe_function(base_type) {
                    self.require_unsafe(expr.span, "call to unsafe function")
                }
            }
            ast::ExprUnary(_, ast::UnDeref, base) => {
                let base_type = ty::node_id_to_type(self.tcx, base.id);
                debug!("effect: unary case, base type is {}",
                        ppaux::ty_to_str(self.tcx, base_type));
                match ty::get(base_type).sty {
                    ty::ty_ptr(_) => {
                        self.require_unsafe(expr.span,
                                            "dereference of unsafe pointer")
                    }
                    _ => {}
                }
            }
            ast::ExprAssign(base, _) | ast::ExprAssignOp(_, _, base, _) => {
                self.check_str_index(base);
            }
            ast::ExprAddrOf(ast::MutMutable, base) => {
                self.check_str_index(base);
            }
            ast::ExprInlineAsm(..) => {
                self.require_unsafe(expr.span, "use of inline assembly")
            }
            ast::ExprPath(..) => {
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
    let mut visitor = EffectCheckVisitor {
        tcx: tcx,
        method_map: method_map,
        unsafe_context: SafeContext,
    };

    visit::walk_crate(&mut visitor, crate, ());
}
