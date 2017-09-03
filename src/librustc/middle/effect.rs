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

use ty::{self, TyCtxt};
use lint;
use lint::builtin::UNUSED_UNSAFE;

use hir::def::Def;
use hir::intravisit::{self, FnKind, Visitor, NestedVisitorMap};
use hir::{self, PatKind};
use syntax::ast;
use syntax_pos::Span;
use util::nodemap::FxHashSet;

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

struct EffectCheckVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    body_id: hir::BodyId,

    /// Whether we're in an unsafe context.
    unsafe_context: UnsafeContext,
    used_unsafe: FxHashSet<ast::NodeId>,
}

impl<'a, 'tcx> EffectCheckVisitor<'a, 'tcx> {
    fn require_unsafe_ext(&mut self, node_id: ast::NodeId, span: Span,
                          description: &str, is_lint: bool) {
        if self.unsafe_context.push_unsafe_count > 0 { return; }
        match self.unsafe_context.root {
            SafeContext => {
                if is_lint {
                    self.tcx.lint_node(lint::builtin::SAFE_EXTERN_STATICS,
                                       node_id,
                                       span,
                                       &format!("{} requires unsafe function or \
                                                 block (error E0133)", description));
                } else {
                    // Report an error.
                    struct_span_err!(
                        self.tcx.sess, span, E0133,
                        "{} requires unsafe function or block", description)
                        .span_label(span, description)
                        .emit();
                }
            }
            UnsafeBlock(block_id) => {
                // OK, but record this.
                debug!("effect: recording unsafe block as used: {}", block_id);
                self.used_unsafe.insert(block_id);
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
        let old_body_id = self.body_id;
        self.tables = self.tcx.body_tables(body);
        self.body_id = body;
        let body = self.tcx.hir.body(body);
        self.visit_body(body);
        self.tables = old_tables;
        self.body_id = old_body_id;
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

        self.unsafe_context = old_unsafe_context;

        // Don't warn about generated blocks, that'll just pollute the output.
        match block.rules {
            hir::UnsafeBlock(hir::UserProvided) => {}
            _ => return,
        }
        if self.used_unsafe.contains(&block.id) {
            return
        }

        /// Return the NodeId for an enclosing scope that is also `unsafe`
        fn is_enclosed(tcx: TyCtxt,
                       used_unsafe: &FxHashSet<ast::NodeId>,
                       id: ast::NodeId) -> Option<(String, ast::NodeId)> {
            let parent_id = tcx.hir.get_parent_node(id);
            if parent_id != id {
                if used_unsafe.contains(&parent_id) {
                    Some(("block".to_string(), parent_id))
                } else if let Some(hir::map::NodeItem(&hir::Item {
                    node: hir::ItemFn(_, hir::Unsafety::Unsafe, _, _, _, _),
                    ..
                })) = tcx.hir.find(parent_id) {
                    Some(("fn".to_string(), parent_id))
                } else {
                    is_enclosed(tcx, used_unsafe, parent_id)
                }
            } else {
                None
            }
        }

        let mut db = self.tcx.struct_span_lint_node(UNUSED_UNSAFE,
                                                    block.id,
                                                    block.span,
                                                    "unnecessary `unsafe` block");
        db.span_label(block.span, "unnecessary `unsafe` block");
        if let Some((kind, id)) = is_enclosed(self.tcx, &self.used_unsafe, block.id) {
            db.span_note(self.tcx.hir.span(id),
                         &format!("because it's nested under this `unsafe` {}", kind));
        }
        db.emit();
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprMethodCall(..) => {
                let def_id = self.tables.type_dependent_defs()[expr.hir_id].def_id();
                let sig = self.tcx.fn_sig(def_id);
                debug!("effect: method call case, signature is {:?}",
                        sig);

                if sig.0.unsafety == hir::Unsafety::Unsafe {
                    self.require_unsafe(expr.span,
                                        "invocation of unsafe method")
                }
            }
            hir::ExprCall(ref base, _) => {
                let base_type = self.tables.expr_ty_adjusted(base);
                debug!("effect: call case, base type is {:?}",
                        base_type);
                match base_type.sty {
                    ty::TyFnDef(..) | ty::TyFnPtr(_) => {
                        if base_type.fn_sig(self.tcx).unsafety() == hir::Unsafety::Unsafe {
                            self.require_unsafe(expr.span, "call to unsafe function")
                        }
                    }
                    _ => {}
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
                    } else if match self.tcx.hir.get_if_local(def_id) {
                        Some(hir::map::NodeForeignItem(..)) => true,
                        Some(..) => false,
                        None => self.tcx.is_foreign_item(def_id),
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
            hir::ExprAssign(ref lhs, ref rhs) => {
                if let hir::ExprField(ref base_expr, field) = lhs.node {
                    if let ty::TyAdt(adt, ..) = self.tables.expr_ty_adjusted(base_expr).sty {
                        if adt.is_union() {
                            let field_ty = self.tables.expr_ty_adjusted(lhs);
                            let owner_def_id = self.tcx.hir.body_owner_def_id(self.body_id);
                            let param_env = self.tcx.param_env(owner_def_id);
                            if field_ty.moves_by_default(self.tcx, param_env, field.span) {
                                self.require_unsafe(field.span,
                                                    "assignment to non-`Copy` union field");
                            }
                            // Do not walk the field expr again.
                            intravisit::walk_expr(self, base_expr);
                            intravisit::walk_expr(self, rhs);
                            return
                        }
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
    let mut visitor = EffectCheckVisitor {
        tcx,
        tables: &ty::TypeckTables::empty(None),
        body_id: hir::BodyId { node_id: ast::CRATE_NODE_ID },
        unsafe_context: UnsafeContext::new(SafeContext),
        used_unsafe: FxHashSet(),
    };

    tcx.hir.krate().visit_all_item_likes(&mut visitor.as_deep_visitor());
}
