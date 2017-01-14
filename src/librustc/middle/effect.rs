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

use dep_graph::DepNode;
use ty::{self, Ty, TyCtxt};
use ty::MethodCall;
use lint;

use syntax::ast;
use syntax_pos::Span;
use hir::{self, PatKind};
use hir::def::Def;
use hir::intravisit::{self, FnKind, Visitor, NestedVisitorMap};

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
        ty::TyFnDef(.., ref f) |
        ty::TyFnPtr(ref f) => f.unsafety == hir::Unsafety::Unsafe,
        _ => false,
    }
}

struct EffectCheckVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::Tables<'tcx>,

    /// Whether we're in an unsafe context.
    unsafe_context: UnsafeContext,
}

impl<'a, 'tcx> EffectCheckVisitor<'a, 'tcx> {
    fn require_unsafe_ext(&mut self, node_id: ast::NodeId, span: Span,
                          description: &str, is_lint: bool) {
        if self.unsafe_context.push_unsafe_count > 0 { return; }
        match self.unsafe_context.root {
            SafeContext => {
                if is_lint {
                    self.tcx.sess.add_lint(lint::builtin::SAFE_EXTERN_STATICS,
                                           node_id,
                                           span,
                                           format!("{} requires unsafe function or \
                                                    block (error E0133)", description));
                } else {
                    // Report an error.
                    struct_span_err!(
                        self.tcx.sess, span, E0133,
                        "{} requires unsafe function or block", description)
                        .span_label(span, &description)
                        .emit();
                }
            }
            UnsafeBlock(block_id) => {
                // OK, but record this.
                debug!("effect: recording unsafe block as used: {}", block_id);
                self.tcx.used_unsafe.borrow_mut().insert(block_id);
            }
            UnsafeFn => {}
        }
    }

    fn require_unsafe(&mut self, span: Span, description: &str) {
        self.require_unsafe_ext(ast::DUMMY_NODE_ID, span, description, false)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for EffectCheckVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.map.body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'tcx>, fn_decl: &'tcx hir::FnDecl,
                body_id: hir::BodyId, span: Span, id: ast::NodeId) {

        let (is_item_fn, is_unsafe_fn) = match fn_kind {
            FnKind::ItemFn(_, _, unsafety, ..) =>
                (true, unsafety == hir::Unsafety::Unsafe),
            FnKind::Method(_, sig, ..) =>
                (true, sig.unsafety == hir::Unsafety::Unsafe),
            _ => (false, false),
        };

        let old_unsafe_context = self.unsafe_context;
        if is_unsafe_fn {
            self.unsafe_context = UnsafeContext::new(UnsafeFn)
        } else if is_item_fn {
            self.unsafe_context = UnsafeContext::new(SafeContext)
        }

        intravisit::walk_fn(self, fn_kind, fn_decl, body_id, span, id);

        self.unsafe_context = old_unsafe_context
    }

    fn visit_block(&mut self, block: &'tcx hir::Block) {
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
            hir::DefaultBlock => {}
        }

        intravisit::walk_block(self, block);

        self.unsafe_context = old_unsafe_context
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprMethodCall(..) => {
                let method_call = MethodCall::expr(expr.id);
                let base_type = self.tables.method_map[&method_call].ty;
                debug!("effect: method call case, base type is {:?}",
                        base_type);
                if type_is_unsafe_function(base_type) {
                    self.require_unsafe(expr.span,
                                        "invocation of unsafe method")
                }
            }
            hir::ExprCall(ref base, _) => {
                let base_type = self.tables.expr_ty_adjusted(base);
                debug!("effect: call case, base type is {:?}",
                        base_type);
                if type_is_unsafe_function(base_type) {
                    self.require_unsafe(expr.span, "call to unsafe function")
                }
            }
            hir::ExprUnary(hir::UnDeref, ref base) => {
                let base_type = self.tables.expr_ty_adjusted(base);
                debug!("effect: unary case, base type is {:?}",
                        base_type);
                if let ty::TyRawPtr(_) = base_type.sty {
                    self.require_unsafe(expr.span, "dereference of raw pointer")
                }
            }
            hir::ExprInlineAsm(..) => {
                self.require_unsafe(expr.span, "use of inline assembly");
            }
            hir::ExprPath(hir::QPath::Resolved(_, ref path)) => {
                if let Def::Static(def_id, mutbl) = path.def {
                    if mutbl {
                        self.require_unsafe(expr.span, "use of mutable static");
                    } else if match self.tcx.map.get_if_local(def_id) {
                        Some(hir::map::NodeForeignItem(..)) => true,
                        Some(..) => false,
                        None => self.tcx.sess.cstore.is_foreign_item(def_id),
                    } {
                        self.require_unsafe_ext(expr.id, expr.span, "use of extern static", true);
                    }
                }
            }
            hir::ExprField(ref base_expr, field) => {
                if let ty::TyAdt(adt, ..) = self.tables.expr_ty_adjusted(base_expr).sty {
                    if adt.is_union() {
                        self.require_unsafe(field.span, "access to union field");
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat) {
        if let PatKind::Struct(_, ref fields, _) = pat.node {
            if let ty::TyAdt(adt, ..) = self.tables.pat_ty(pat).sty {
                if adt.is_union() {
                    for field in fields {
                        self.require_unsafe(field.span, "matching on union field");
                    }
                }
            }
        }

        intravisit::walk_pat(self, pat);
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let _task = tcx.dep_graph.in_task(DepNode::EffectCheck);

    let mut visitor = EffectCheckVisitor {
        tcx: tcx,
        tables: &ty::Tables::empty(),
        unsafe_context: UnsafeContext::new(SafeContext),
    };

    tcx.map.krate().visit_all_item_likes(&mut visitor.as_deep_visitor());
}
