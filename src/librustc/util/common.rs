// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use syntax::ast;
use syntax::visit;
use syntax::visit::Visitor;

use time;

pub struct Timer {
    start: f64,
    what: &'static str,
    enabled: bool,
    old_depth: uint,
}

local_data_key!(depth: uint)

impl Timer {
    pub fn new(enabled: bool, what: &'static str) -> Timer {
        let old_depth = depth.get().map(|d| *d).unwrap_or(0);
        depth.replace(Some(old_depth + 1));

        Timer {
            start: time::precise_time_s(),
            what: what,
            enabled: enabled,
            old_depth: old_depth,
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let end = time::precise_time_s();
        depth.replace(Some(self.old_depth));
        println!("{}time: {:3.3f} s\t{}",
                 "  ".repeat(self.old_depth),
                 end - self.start,
                 self.what);
    }
}

pub fn indent<R>(op: || -> R) -> R {
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = {:?})", r);
    r
}

pub struct _indenter {
    _i: (),
}

impl Drop for _indenter {
    fn drop(&mut self) { debug!("<<"); }
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

struct LoopQueryVisitor<'a> {
    p: |&ast::Expr_|: 'a -> bool,
    flag: bool,
}

impl<'a> Visitor<()> for LoopQueryVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr, _: ()) {
        self.flag |= (self.p)(&e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::ExprLoop(..) | ast::ExprWhile(..) => {}
          _ => visit::walk_expr(self, e, ())
        }
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn loop_query(b: &ast::Block, p: |&ast::Expr_| -> bool) -> bool {
    let mut v = LoopQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, b, ());
    return v.flag;
}

struct BlockQueryVisitor<'a> {
    p: |&ast::Expr|: 'a -> bool,
    flag: bool,
}

impl<'a> Visitor<()> for BlockQueryVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr, _: ()) {
        self.flag |= (self.p)(e);
        visit::walk_expr(self, e, ())
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn block_query(b: ast::P<ast::Block>, p: |&ast::Expr| -> bool) -> bool {
    let mut v = BlockQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, b, ());
    return v.flag;
}
