// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use syntax::ast;
use syntax::codemap::{span};
use syntax::visit;
use syntax::print;
use syntax;

use core::option;
use core::str;
use core::vec;
use std::map::HashMap;

fn indent<R>(op: fn() -> R) -> R {
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = %?)", r);
    move r
}

struct _indenter {
    _i: (),
    drop { debug!("<<"); }
}

fn _indenter(_i: ()) -> _indenter {
    _indenter {
        _i: ()
    }
}

fn indenter() -> _indenter {
    debug!(">>");
    _indenter(())
}

type flag = HashMap<~str, ()>;

fn field_expr(f: ast::field) -> @ast::expr { return f.node.expr; }

fn field_exprs(fields: ~[ast::field]) -> ~[@ast::expr] {
    fields.map(|f| f.node.expr)
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
fn loop_query(b: ast::blk, p: fn@(ast::expr_) -> bool) -> bool {
    let rs = @mut false;
    let visit_expr: @fn(@ast::expr,
                        &&flag: @mut bool,
                        v: visit::vt<@mut bool>) = |e, &&flag, v| {
        *flag |= p(e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::expr_loop(*) | ast::expr_while(*)
          | ast::expr_loop_body(*) => {}
          _ => visit::visit_expr(e, flag, v)
        }
    };
    let v = visit::mk_vt(@visit::Visitor {
        visit_expr: visit_expr,
        .. *visit::default_visitor()});
    visit::visit_block(b, rs, v);
    return *rs;
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
fn block_query(b: ast::blk, p: fn@(@ast::expr) -> bool) -> bool {
    let rs = @mut false;
    let visit_expr: @fn(@ast::expr,
                        &&flag: @mut bool,
                        v: visit::vt<@mut bool>) = |e, &&flag, v| {
        *flag |= p(e);
        visit::visit_expr(e, flag, v)
    };
    let v = visit::mk_vt(@visit::Visitor{
        visit_expr: visit_expr,
        .. *visit::default_visitor()});
    visit::visit_block(b, rs, v);
    return *rs;
}

fn local_rhs_span(l: @ast::local, def: span) -> span {
    match l.node.init {
      Some(i) => return i.span,
      _ => return def
    }
}

fn pluralize(n: uint, +s: ~str) -> ~str {
    if n == 1 { s }
    else { str::concat([s, ~"s"]) }
}

// A set of node IDs (used to keep track of which node IDs are for statements)
type stmt_set = HashMap<ast::node_id, ()>;

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
