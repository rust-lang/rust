// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use std::cell::{RefCell, Cell};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
#[cfg(stage0)] use std::hash::Hasher;
use std::iter::repeat;
use std::time::Duration;
use std::collections::hash_state::HashState;

use syntax::ast;
use syntax::visit;
use syntax::visit::Visitor;

// The name of the associated type for `Fn` return types
pub const FN_OUTPUT_NAME: &'static str = "Output";

// Useful type to use with `Result<>` indicate that an error has already
// been reported to the user, so no need to continue checking.
#[derive(Clone, Copy, Debug)]
pub struct ErrorReported;

pub fn time<T, U, F>(do_it: bool, what: &str, u: U, f: F) -> T where
    F: FnOnce(U) -> T,
{
    thread_local!(static DEPTH: Cell<uint> = Cell::new(0));
    if !do_it { return f(u); }

    let old = DEPTH.with(|slot| {
        let r = slot.get();
        slot.set(r + 1);
        r
    });

    let mut u = Some(u);
    let mut rv = None;
    let dur = {
        let ref mut rvp = rv;

        Duration::span(move || {
            *rvp = Some(f(u.take().unwrap()))
        })
    };
    let rv = rv.unwrap();

    println!("{}time: {}.{:03} \t{}", repeat("  ").take(old).collect::<String>(),
             dur.num_seconds(), dur.num_milliseconds() % 1000, what);
    DEPTH.with(|slot| slot.set(old));

    rv
}

pub fn indent<R, F>(op: F) -> R where
    R: Debug,
    F: FnOnce() -> R,
{
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

struct LoopQueryVisitor<P> where P: FnMut(&ast::Expr_) -> bool {
    p: P,
    flag: bool,
}

impl<'v, P> Visitor<'v> for LoopQueryVisitor<P> where P: FnMut(&ast::Expr_) -> bool {
    fn visit_expr(&mut self, e: &ast::Expr) {
        self.flag |= (self.p)(&e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          ast::ExprLoop(..) | ast::ExprWhile(..) => {}
          _ => visit::walk_expr(self, e)
        }
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn loop_query<P>(b: &ast::Block, p: P) -> bool where P: FnMut(&ast::Expr_) -> bool {
    let mut v = LoopQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, b);
    return v.flag;
}

struct BlockQueryVisitor<P> where P: FnMut(&ast::Expr) -> bool {
    p: P,
    flag: bool,
}

impl<'v, P> Visitor<'v> for BlockQueryVisitor<P> where P: FnMut(&ast::Expr) -> bool {
    fn visit_expr(&mut self, e: &ast::Expr) {
        self.flag |= (self.p)(e);
        visit::walk_expr(self, e)
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn block_query<P>(b: &ast::Block, p: P) -> bool where P: FnMut(&ast::Expr) -> bool {
    let mut v = BlockQueryVisitor {
        p: p,
        flag: false,
    };
    visit::walk_block(&mut v, &*b);
    return v.flag;
}

/// K: Eq + Hash<S>, V, S, H: Hasher<S>
///
/// Determines whether there exists a path from `source` to `destination`.  The graph is defined by
/// the `edges_map`, which maps from a node `S` to a list of its adjacent nodes `T`.
///
/// Efficiency note: This is implemented in an inefficient way because it is typically invoked on
/// very small graphs. If the graphs become larger, a more efficient graph representation and
/// algorithm would probably be advised.
#[cfg(stage0)]
pub fn can_reach<T, S>(edges_map: &HashMap<T, Vec<T>, S>, source: T,
                       destination: T) -> bool
    where S: HashState,
          <S as HashState>::Hasher: Hasher<Output=u64>,
          T: Hash<<S as HashState>::Hasher> + Eq + Clone,
{
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
        match edges_map.get(&queue[i]) {
            Some(edges) => {
                for target in edges {
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
/// K: Eq + Hash<S>, V, S, H: Hasher<S>
///
/// Determines whether there exists a path from `source` to `destination`.  The graph is defined by
/// the `edges_map`, which maps from a node `S` to a list of its adjacent nodes `T`.
///
/// Efficiency note: This is implemented in an inefficient way because it is typically invoked on
/// very small graphs. If the graphs become larger, a more efficient graph representation and
/// algorithm would probably be advised.
#[cfg(not(stage0))]
pub fn can_reach<T, S>(edges_map: &HashMap<T, Vec<T>, S>, source: T,
                       destination: T) -> bool
    where S: HashState, T: Hash + Eq + Clone,
{
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
        match edges_map.get(&queue[i]) {
            Some(edges) => {
                for target in edges {
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

/// Memoizes a one-argument closure using the given RefCell containing
/// a type implementing MutableMap to serve as a cache.
///
/// In the future the signature of this function is expected to be:
/// ```
/// pub fn memoized<T: Clone, U: Clone, M: MutableMap<T, U>>(
///    cache: &RefCell<M>,
///    f: &|T| -> U
/// ) -> impl |T| -> U {
/// ```
/// but currently it is not possible.
///
/// # Example
/// ```
/// struct Context {
///    cache: RefCell<HashMap<uint, uint>>
/// }
///
/// fn factorial(ctxt: &Context, n: uint) -> uint {
///     memoized(&ctxt.cache, n, |n| match n {
///         0 | 1 => n,
///         _ => factorial(ctxt, n - 2) + factorial(ctxt, n - 1)
///     })
/// }
/// ```
#[inline(always)]
#[cfg(stage0)]
pub fn memoized<T, U, S, F>(cache: &RefCell<HashMap<T, U, S>>, arg: T, f: F) -> U
    where T: Clone + Hash<<S as HashState>::Hasher> + Eq,
          U: Clone,
          S: HashState,
          <S as HashState>::Hasher: Hasher<Output=u64>,
          F: FnOnce(T) -> U,
{
    let key = arg.clone();
    let result = cache.borrow().get(&key).cloned();
    match result {
        Some(result) => result,
        None => {
            let result = f(arg);
            cache.borrow_mut().insert(key, result.clone());
            result
        }
    }
}
/// Memoizes a one-argument closure using the given RefCell containing
/// a type implementing MutableMap to serve as a cache.
///
/// In the future the signature of this function is expected to be:
/// ```
/// pub fn memoized<T: Clone, U: Clone, M: MutableMap<T, U>>(
///    cache: &RefCell<M>,
///    f: &|T| -> U
/// ) -> impl |T| -> U {
/// ```
/// but currently it is not possible.
///
/// # Example
/// ```
/// struct Context {
///    cache: RefCell<HashMap<uint, uint>>
/// }
///
/// fn factorial(ctxt: &Context, n: uint) -> uint {
///     memoized(&ctxt.cache, n, |n| match n {
///         0 | 1 => n,
///         _ => factorial(ctxt, n - 2) + factorial(ctxt, n - 1)
///     })
/// }
/// ```
#[inline(always)]
#[cfg(not(stage0))]
pub fn memoized<T, U, S, F>(cache: &RefCell<HashMap<T, U, S>>, arg: T, f: F) -> U
    where T: Clone + Hash + Eq,
          U: Clone,
          S: HashState,
          F: FnOnce(T) -> U,
{
    let key = arg.clone();
    let result = cache.borrow().get(&key).map(|result| result.clone());
    match result {
        Some(result) => result,
        None => {
            let result = f(arg);
            cache.borrow_mut().insert(key, result.clone());
            result
        }
    }
}
