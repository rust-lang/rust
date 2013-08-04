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
use syntax::codemap::{span};
use syntax::oldvisit;

use std::hashmap::HashSet;
use extra;

pub fn time<T>(do_it: bool, what: ~str, thunk: &fn() -> T) -> T {
    if !do_it { return thunk(); }
    let start = extra::time::precise_time_s();
    let rv = thunk();
    let end = extra::time::precise_time_s();
    printfln!("time: %3.3f s\t%s", end - start, what);
    rv
}

pub fn indent<R>(op: &fn() -> R) -> R {
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = %?)", r);
    r
}

pub struct _indenter {
    _i: (),
}

impl Drop for _indenter {
    fn drop(&self) { debug!("<<"); }
}

pub fn _indenter(_i: ()) -> _indenter {
    _indenter {
        _i: ()
    }
}

pub fn indenter() -> _indenter {
    debug!(">>");
    _indenter(())
}

pub fn field_expr(f: ast::Field) -> @ast::expr { return f.expr; }

pub fn field_exprs(fields: ~[ast::Field]) -> ~[@ast::expr] {
    fields.map(|f| f.expr)
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn loop_query(b: &ast::Block, p: @fn(&ast::expr_) -> bool) -> bool {
    let rs = @mut false;
    let visit_expr: @fn(@ast::expr,
                        (@mut bool,
                         oldvisit::vt<@mut bool>)) = |e, (flag, v)| {
        *flag |= p(&e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::expr_loop(*) | ast::expr_while(*) => {}
          _ => oldvisit::visit_expr(e, (flag, v))
        }
    };
    let v = oldvisit::mk_vt(@oldvisit::Visitor {
        visit_expr: visit_expr,
        .. *oldvisit::default_visitor()});
    oldvisit::visit_block(b, (rs, v));
    return *rs;
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn block_query(b: &ast::Block, p: @fn(@ast::expr) -> bool) -> bool {
    let rs = @mut false;
    let visit_expr: @fn(@ast::expr,
                        (@mut bool,
                         oldvisit::vt<@mut bool>)) = |e, (flag, v)| {
        *flag |= p(e);
        oldvisit::visit_expr(e, (flag, v))
    };
    let v = oldvisit::mk_vt(@oldvisit::Visitor{
        visit_expr: visit_expr,
        .. *oldvisit::default_visitor()});
    oldvisit::visit_block(b, (rs, v));
    return *rs;
}

pub fn local_rhs_span(l: @ast::Local, def: span) -> span {
    match l.init {
      Some(i) => return i.span,
      _ => return def
    }
}

pub fn pluralize(n: uint, s: ~str) -> ~str {
    if n == 1 { s }
    else { fmt!("%ss", s) }
}

// A set of node IDs (used to keep track of which node IDs are for statements)
pub type stmt_set = @mut HashSet<ast::NodeId>;
