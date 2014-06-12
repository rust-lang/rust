// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

extern crate collections;
extern crate time;

use time::precise_time_s;
use std::os;
use std::task;
use std::vec;
use std::gc::{Gc, GC};

#[deriving(Clone)]
enum List<T> {
    Nil, Cons(T, Gc<List<T>>)
}

enum UniqueList {
    ULNil, ULCons(Box<UniqueList>)
}

fn main() {
    let (repeat, depth) = if os::getenv("RUST_BENCH").is_some() {
        (50, 1000)
    } else {
        (10, 10)
    };

    run(repeat, depth);
}

fn run(repeat: int, depth: int) {
    for _ in range(0, repeat) {
        println!("starting {:.4f}", precise_time_s());
        task::try(proc() {
            recurse_or_fail(depth, None)
        });
        println!("stopping {:.4f}", precise_time_s());
    }
}

type nillist = List<()>;

// Filled with things that have to be unwound

struct State {
    managed: Gc<nillist>,
    unique: Box<nillist>,
    tuple: (Gc<nillist>, Box<nillist>),
    vec: Vec<Gc<nillist>>,
    res: r
}

struct r {
  _l: Gc<nillist>,
}

#[unsafe_destructor]
impl Drop for r {
    fn drop(&mut self) {}
}

fn r(l: Gc<nillist>) -> r {
    r {
        _l: l
    }
}

fn recurse_or_fail(depth: int, st: Option<State>) {
    if depth == 0 {
        println!("unwinding {:.4f}", precise_time_s());
        fail!();
    } else {
        let depth = depth - 1;

        let st = match st {
          None => {
            State {
                managed: box(GC) Nil,
                unique: box Nil,
                tuple: (box(GC) Nil, box Nil),
                vec: vec!(box(GC) Nil),
                res: r(box(GC) Nil)
            }
          }
          Some(st) => {
            State {
                managed: box(GC) Cons((), st.managed),
                unique: box Cons((), box(GC) *st.unique),
                tuple: (box(GC) Cons((), st.tuple.ref0().clone()),
                        box Cons((), box(GC) *st.tuple.ref1().clone())),
                vec: st.vec.clone().append(
                        &[box(GC) Cons((), *st.vec.last().unwrap())]),
                res: r(box(GC) Cons((), st.res._l))
            }
          }
        };

        recurse_or_fail(depth, Some(st));
    }
}
