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

use std::hash::{Hash, Hasher};
use std::collections::HashMap;
use syntax::ast;
use syntax::visit;
use syntax::visit::Visitor;

use time;

pub fn time<T, U>(do_it: bool, what: &str, u: U, f: |U| -> T) -> T {
    local_data_key!(depth: uint);
    if !do_it { return f(u); }

    let old = depth.get().map(|d| *d).unwrap_or(0);
    depth.replace(Some(old + 1));

    let start = time::precise_time_s();
    let rv = f(u);
    let end = time::precise_time_s();

    println!("{}time: {:3.3f} s\t{}", "  ".repeat(old), end - start, what);
    depth.replace(Some(old));

    rv
}

pub fn indent<R>(op: || -> R) -> R {
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = {:?})", r);
    r
}

pub struct Indenter {
    _cannot_construct_outside_of_this_module: ()
}

impl Drop for Indenter {
    fn drop(&mut self) { debug!("<<"); }
}

pub fn indenter() -> Indenter {
    debug!(">>");
    Indenter { _cannot_construct_outside_of_this_module: () }
}

struct LoopQueryVisitor<'a> {
    p: |&ast::Expr_|: 'a -> bool,
    flag: bool,
}

impl<'a, 'v> Visitor<'v> for LoopQueryVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr) {
        self.flag |= (self.p)(&e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::ExprLoop(..) | ast::ExprWhile(..) | ast::ExprForLoop(..) => {}
          _ => visit::walk_expr(self, e)
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
    visit::walk_block(&mut v, b);
    return v.flag;
}

struct BlockQueryVisitor<'a> {
    p: |&ast::Expr|: 'a -> bool,
    flag: bool,
}

impl<'a, 'v> Visitor<'v> for BlockQueryVisitor<'a> {
    fn visit_expr(&mut self, e: &ast::Expr) {
        self.flag |= (self.p)(e);
        visit::walk_expr(self, e)
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn block_query(b: ast::P<ast::Block>, p: |&ast::Expr| -> bool) -> bool {
    let mut v = BlockQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, &*b);
    return v.flag;
}

// K: Eq + Hash<S>, V, S, H: Hasher<S>
pub fn can_reach<S,H:Hasher<S>,T:Eq+Clone+Hash<S>>(
    edges_map: &HashMap<T,Vec<T>,H>,
    source: T,
    destination: T)
    -> bool
{
    /*!
     * Determines whether there exists a path from `source` to
     * `destination`.  The graph is defined by the `edges_map`, which
     * maps from a node `S` to a list of its adjacent nodes `T`.
     *
     * Efficiency note: This is implemented in an inefficient way
     * because it is typically invoked on very small graphs. If the graphs
     * become larger, a more efficient graph representation and algorithm
     * would probably be advised.
     */

    if source == destination {
        return true;
    }

    // Do a little breadth-first-search here.  The `queue` list
    // doubles as a way to detect if we've seen a particular FR
    // before.  Note that we expect this graph to be an *extremely
    // shallow* tree.
    let mut queue = vec!(source);
    let mut i = 0;
    while i < queue.len() {
        match edges_map.find(queue.get(i)) {
            Some(edges) => {
                for target in edges.iter() {
                    if *target == destination {
                        return true;
                    }

                    if !queue.iter().any(|x| x == target) {
                        queue.push((*target).clone());
                    }
                }
            }
            None => {}
        }
        i += 1;
    }
    return false;
}
