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
//! `unsafe`.
use self::RootUnsafeContext::*;

use middle::def;
use middle::ty::{self, Ty};
use middle::ty::MethodCall;

use syntax::ast;
use syntax::codemap::Span;
use rustc_front::hir;
use rustc_front::intravisit;
use rustc_front::intravisit::{FnKind, Visitor};

#[derive(Copy, Clone)]
struct UnsafeContext {
    push_unsafe_count: usize,
    root: RootUnsafeContext,
}

impl UnsafeContext {
    fn new(root: RootUnsafeContext) -> UnsafeContext {
        UnsafeContext { root: root, push_unsafe_count: 0 }
    }
}

#[derive(Copy, Clone, PartialEq)]
enum RootUnsafeContext {
    SafeContext,
    UnsafeFn,
    UnsafeBlock(ast::NodeId),
}

fn type_is_unsafe_function(ty: Ty) -> bool {
    match ty.sty {
        ty::TyBareFn(_, ref f) => f.unsafety == hir::Unsafety::Unsafe,
        _ => false,
    }
}

struct EffectCheckVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,

    /// Whether we're in an unsafe context.
    unsafe_context: UnsafeContext,
}

impl<'a, 'tcx> EffectCheckVisitor<'a, 'tcx> {
    fn require_unsafe(&mut self, span: Span, description: &str) {
        if self.unsafe_context.push_unsafe_count > 0 { return; }
        match self.unsafe_context.root {
            SafeContext => {
                // Report an error.
                span_err!(self.tcx.sess, span, E0133,
                          "{} requires unsafe function or block",
                          description);
            }
            UnsafeBlock(block_id) => {
                // OK, but record this.
                debug!("effect: recording unsafe block as used: {}", block_id);
                self.tcx.used_unsafe.borrow_mut().insert(block_id);
            }
            UnsafeFn => {}
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for EffectCheckVisitor<'a, 'tcx> {
    fn visit_fn(&mut self, fn_kind: FnKind<'v>, fn_decl: &'v hir::FnDecl,
                block: &'v hir::Block, span: Span, _: ast::NodeId) {

        let (is_item_fn, is_unsafe_fn) = match fn_kind {
            FnKind::ItemFn(_, _, unsafety, _, _, _) =>
                (true, unsafety == hir::Unsafety::Unsafe),
            FnKind::Method(_, sig, _) =>
                (true, sig.unsafety == hir::Unsafety::Unsafe),
            _ => (false, false),
        };

        let old_unsafe_context = self.unsafe_context;
        if is_unsafe_fn {
            self.unsafe_context = UnsafeContext::new(UnsafeFn)
        } else if is_item_fn {
            self.unsafe_context = UnsafeContext::new(SafeContext)
        }

        intravisit::walk_fn(self, fn_kind, fn_decl, block, span);

        self.unsafe_context = old_unsafe_context
    }

    fn visit_block(&mut self, block: &hir::Block) {
        let old_unsafe_context = self.unsafe_context;
        match block.rules {
            hir::UnsafeBlock(source) => {
                // By default only the outermost `unsafe` block is
                // "used" and so nested unsafe blocks are pointless
                // (the inner ones are unnecessary and we actually
                // warn about them). As such, there are two cases when
                // we need to create a new context, when we're
                // - outside `unsafe` and found a `unsafe` block
                //   (normal case)
                // - inside `unsafe`, found an `unsafe` block
                //   created internally to the compiler
                //
                // The second case is necessary to ensure that the
                // compiler `unsafe` blocks don't accidentally "use"
                // external blocks (e.g. `unsafe { println("") }`,
                // expands to `unsafe { ... unsafe { ... } }` where
                // the inner one is compiler generated).
                if self.unsafe_context.root == SafeContext || source == hir::CompilerGenerated {
                    self.unsafe_context.root = UnsafeBlock(block.id)
                }
            }
            hir::PushUnsafeBlock(..) => {
                self.unsafe_context.push_unsafe_count =
                    self.unsafe_context.push_unsafe_count.checked_add(1).unwrap();
            }
            hir::PopUnsafeBlock(..) => {
                self.unsafe_context.push_unsafe_count =
                    self.unsafe_context.push_unsafe_count.checked_sub(1).unwrap();
            }
            hir::DefaultBlock | hir::PushUnstableBlock | hir:: PopUnstableBlock => {}
        }

        intravisit::walk_block(self, block);

        self.unsafe_context = old_unsafe_context
    }

    fn visit_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprMethodCall(_, _, _) => {
                let method_call = MethodCall::expr(expr.id);
                let base_type = self.tcx.tables.borrow().method_map[&method_call].ty;
                debug!("effect: method call case, base type is {:?}",
                        base_type);
                if type_is_unsafe_function(base_type) {
                    self.require_unsafe(expr.span,
                                        "invocation of unsafe method")
                }
            }
            hir::ExprCall(ref base, _) => {
                let base_type = self.tcx.expr_ty_adjusted(base);
                debug!("effect: call case, base type is {:?}",
                        base_type);
                if type_is_unsafe_function(base_type) {
                    self.require_unsafe(expr.span, "call to unsafe function")
                }
            }
            hir::ExprUnary(hir::UnDeref, ref base) => {
                let base_type = self.tcx.expr_ty_adjusted(base);
                debug!("effect: unary case, base type is {:?}",
                        base_type);
                if let ty::TyRawPtr(_) = base_type.sty {
                    self.require_unsafe(expr.span, "dereference of raw pointer")
                }
            }
            hir::ExprInlineAsm(..) => {
                self.require_unsafe(expr.span, "use of inline assembly");
            }
            hir::ExprPath(..) => {
                if let def::DefStatic(_, true) = self.tcx.resolve_expr(expr) {
                    self.require_unsafe(expr.span, "use of mutable static");
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut visitor = EffectCheckVisitor {
        tcx: tcx,
        unsafe_context: UnsafeContext::new(SafeContext),
    };

    tcx.map.krate().visit_all_items(&mut visitor);
}
