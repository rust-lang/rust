// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use syntax::ast;
use syntax::codemap::{Span};
use syntax::visit;
use syntax::visit::Visitor;

use std::hashmap::HashSet;
use extra;

pub fn time<T, U>(do_it: bool, what: ~str, u: U, f: &fn(U) -> T) -> T {
    if !do_it { return f(u); }
    let start = extra::time::precise_time_s();
    let rv = f(u);
    let end = extra::time::precise_time_s();
    println!("time: {:3.3f} s\t{}", end - start, what);
    rv
}

pub fn indent<R>(op: &fn() -> R) -> R {
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug2!(">>");
    let r = op();
    debug2!("<< (Result = {:?})", r);
    r
}

pub struct _indenter {
    _i: (),
}

impl Drop for _indenter {
    fn drop(&mut self) { debug2!("<<"); }
}

pub fn _indenter(_i: ()) -> _indenter {
    _indenter {
        _i: ()
    }
}

pub fn indenter() -> _indenter {
    debug2!(">>");
    _indenter(())
}

pub fn field_expr(f: ast::Field) -> @ast::Expr { return f.expr; }

pub fn field_exprs(fields: ~[ast::Field]) -> ~[@ast::Expr] {
    fields.map(|f| f.expr)
}

struct LoopQueryVisitor<'self> {
    p: &'self fn(&ast::Expr_) -> bool,
    flag: bool,
}

impl<'self> Visitor<()> for LoopQueryVisitor<'self> {
    fn visit_expr(&mut self, e: @ast::Expr, _: ()) {
        self.flag |= (self.p)(&e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::ExprLoop(*) | ast::ExprWhile(*) => {}
          _ => visit::walk_expr(self, e, ())
        }
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn loop_query(b: &ast::Block, p: &fn(&ast::Expr_) -> bool) -> bool {
    let mut v = LoopQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, b, ());
    return v.flag;
}

struct BlockQueryVisitor<'self> {
    p: &'self fn(@ast::Expr) -> bool,
    flag: bool,
}

impl<'self> Visitor<()> for BlockQueryVisitor<'self> {
    fn visit_expr(&mut self, e: @ast::Expr, _:()) {
        self.flag |= (self.p)(e);
        visit::walk_expr(self, e, ())
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn block_query(b: &ast::Block, p: &fn(@ast::Expr) -> bool) -> bool {
    let mut v = BlockQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, b, ());
    return v.flag;
}

pub fn local_rhs_span(l: @ast::Local, def: Span) -> Span {
    match l.init {
      Some(i) => return i.span,
      _ => return def
    }
}

pub fn pluralize(n: uint, s: ~str) -> ~str {
    if n == 1 { s }
    else { format!("{}s", s) }
}

// A set of node IDs (used to keep track of which node IDs are for statements)
pub type stmt_set = @mut HashSet<ast::NodeId>;
