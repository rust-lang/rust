// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use self::Context::*;

use rustc::session::Session;

use rustc::dep_graph::DepNode;
use rustc::hir::map::Map;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir;
use syntax::ast;
use syntax_pos::Span;

#[derive(Clone, Copy, PartialEq)]
enum LoopKind {
    Loop(hir::LoopSource),
    WhileLoop,
}

impl LoopKind {
    fn name(self) -> &'static str {
        match self {
            LoopKind::Loop(hir::LoopSource::Loop) => "loop",
            LoopKind::Loop(hir::LoopSource::WhileLet) => "while let",
            LoopKind::Loop(hir::LoopSource::ForLoop) => "for",
            LoopKind::WhileLoop => "while",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Context {
    Normal,
    Loop(LoopKind),
    Closure,
}

#[derive(Copy, Clone)]
struct CheckLoopVisitor<'a, 'ast: 'a> {
    sess: &'a Session,
    hir_map: &'a Map<'ast>,
    cx: Context,
}

pub fn check_crate(sess: &Session, map: &Map) {
    let _task = map.dep_graph.in_task(DepNode::CheckLoops);
    let krate = map.krate();
    krate.visit_all_item_likes(&mut CheckLoopVisitor {
        sess: sess,
        hir_map: map,
        cx: Normal,
    }.as_deep_visitor());
}

impl<'a, 'ast> Visitor<'ast> for CheckLoopVisitor<'a, 'ast> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'ast> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_item(&mut self, i: &'ast hir::Item) {
        self.with_context(Normal, |v| intravisit::walk_item(v, i));
    }

    fn visit_impl_item(&mut self, i: &'ast hir::ImplItem) {
        self.with_context(Normal, |v| intravisit::walk_impl_item(v, i));
    }

    fn visit_expr(&mut self, e: &'ast hir::Expr) {
        match e.node {
            hir::ExprWhile(ref e, ref b, _) => {
                self.with_context(Loop(LoopKind::WhileLoop), |v| {
                    v.visit_expr(&e);
                    v.visit_block(&b);
                });
            }
            hir::ExprLoop(ref b, _, source) => {
                self.with_context(Loop(LoopKind::Loop(source)), |v| v.visit_block(&b));
            }
            hir::ExprClosure(.., b, _) => {
                self.with_context(Closure, |v| v.visit_nested_body(b));
            }
            hir::ExprBreak(label, ref opt_expr) => {
                if opt_expr.is_some() {
                    let loop_kind = if let Some(label) = label {
                        if label.loop_id == ast::DUMMY_NODE_ID {
                            None
                        } else {
                            Some(match self.hir_map.expect_expr(label.loop_id).node {
                                hir::ExprWhile(..) => LoopKind::WhileLoop,
                                hir::ExprLoop(_, _, source) => LoopKind::Loop(source),
                                ref r => span_bug!(e.span,
                                                   "break label resolved to a non-loop: {:?}", r),
                            })
                        }
                    } else if let Loop(kind) = self.cx {
                        Some(kind)
                    } else {
                        // `break` outside a loop - caught below
                        None
                    };
                    match loop_kind {
                        None | Some(LoopKind::Loop(hir::LoopSource::Loop)) => (),
                        Some(kind) => {
                            struct_span_err!(self.sess, e.span, E0571,
                                             "`break` with value from a `{}` loop",
                                             kind.name())
                                .span_label(e.span,
                                            &format!("can only break with a value inside `loop`"))
                                .emit();
                        }
                    }
                }
                self.require_loop("break", e.span);
            }
            hir::ExprAgain(_) => self.require_loop("continue", e.span),
            _ => intravisit::walk_expr(self, e),
        }
    }
}

impl<'a, 'ast> CheckLoopVisitor<'a, 'ast> {
    fn with_context<F>(&mut self, cx: Context, f: F)
        where F: FnOnce(&mut CheckLoopVisitor<'a, 'ast>)
    {
        let old_cx = self.cx;
        self.cx = cx;
        f(self);
        self.cx = old_cx;
    }

    fn require_loop(&self, name: &str, span: Span) {
        match self.cx {
            Loop(_) => {}
            Closure => {
                struct_span_err!(self.sess, span, E0267, "`{}` inside of a closure", name)
                .span_label(span, &format!("cannot break inside of a closure"))
                .emit();
            }
            Normal => {
                struct_span_err!(self.sess, span, E0268, "`{}` outside of loop", name)
                .span_label(span, &format!("cannot break outside of a loop"))
                .emit();
            }
        }
    }
}
