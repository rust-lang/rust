// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::cast;
use std::num;
use std::ptr;
use std::sys;
use std::vec;
use future_spawn = future::spawn;

/**
 * The maximum number of tasks this module will spawn for a single
 * operation.
 */
static MAX_TASKS : uint = 32u;

/// The minimum number of elements each task will process.
static MIN_GRANULARITY : uint = 1024u;

/**
 * An internal helper to map a function over a large vector and
 * return the intermediate results.
 *
 * This is used to build most of the other parallel vector functions,
 * like map or alli.
 */
fn map_slices<A:Clone + Send,B:Clone + Send>(
    xs: &[A],
    f: &fn() -> ~fn(uint, v: &[A]) -> B)
    -> ~[B] {

    let len = xs.len();
    if len < MIN_GRANULARITY {
        info!("small slice");
        // This is a small vector, fall back on the normal map.
        ~[f()(0u, xs)]
    } else {
        let num_tasks = num::min(MAX_TASKS, len / MIN_GRANULARITY);

        let items_per_task = len / num_tasks;

        let mut futures = ~[];
        let mut base = 0u;
        info!("spawning tasks");
        while base < len {
            let end = num::min(len, base + items_per_task);
            do xs.as_imm_buf |p, _len| {
                let f = f();
                let base = base;
                let f = do future_spawn() || {
                    unsafe {
                        let len = end - base;
                        let slice = (ptr::offset(p, base as int),
                                     len * sys::size_of::<A>());
                        info!("pre-slice: %?", (base, slice));
                        let slice : &[A] =
                            cast::transmute(slice);
                        info!("slice: %?", (base, slice.len(), end - base));
                        assert_eq!(slice.len(), end - base);
                        f(base, slice)
                    }
                };
                futures.push(f);
            };
            base += items_per_task;
        }
        info!("tasks spawned");

        info!("num_tasks: %?", (num_tasks, futures.len()));
        assert_eq!(num_tasks, futures.len());

        do futures.move_iter().map |ys| {
            let mut ys = ys;
            ys.get()
        }.collect()
    }
}

/// A parallel version of map.
pub fn map<A:Clone + Send,B:Clone + Send>(
    xs: &[A], fn_factory: &fn() -> ~fn(&A) -> B) -> ~[B] {
    vec::concat(map_slices(xs, || {
        let f = fn_factory();
        let result: ~fn(uint, &[A]) -> ~[B] =
            |_, slice| slice.iter().map(|x| f(x)).collect();
        result
    }))
}

/// A parallel version of mapi.
pub fn mapi<A:Clone + Send,B:Clone + Send>(
        xs: &[A],
        fn_factory: &fn() -> ~fn(uint, &A) -> B) -> ~[B] {
    let slices = map_slices(xs, || {
        let f = fn_factory();
        let result: ~fn(uint, &[A]) -> ~[B] = |base, slice| {
            slice.iter().enumerate().map(|(i, x)| {
                f(i + base, x)
            }).collect()
        };
        result
    });
    let r = vec::concat(slices);
    info!("%?", (r.len(), xs.len()));
    assert_eq!(r.len(), xs.len());
    r
}

/// Returns true if the function holds for all elements in the vector.
pub fn alli<A:Clone + Send>(
    xs: &[A],
    fn_factory: &fn() -> ~fn(uint, &A) -> bool) -> bool
{
    let mapped = map_slices(xs, || {
        let f = fn_factory();
        let result: ~fn(uint, &[A]) -> bool = |base, slice| {
            slice.iter().enumerate().all(|(i, x)| f(i + base, x))
        };
        result
    });
    mapped.iter().all(|&x| x)
}

/// Returns true if the function holds for any elements in the vector.
pub fn any<A:Clone + Send>(
    xs: &[A],
    fn_factory: &fn() -> ~fn(&A) -> bool) -> bool {
    let mapped = map_slices(xs, || {
        let f = fn_factory();
        let result: ~fn(uint, &[A]) -> bool = |_, slice| slice.iter().any(f);
        result
    });
    mapped.iter().any(|&x| x)
}
