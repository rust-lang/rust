// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

extern crate collections;
extern crate time;

use time::precise_time_s;
use std::os;
use std::task;
use std::vec;

#[deriving(Clone)]
enum List<T> {
    Nil, Cons(T, @List<T>)
}

enum UniqueList {
    ULNil, ULCons(~UniqueList)
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
    managed: @nillist,
    unique: ~nillist,
    tuple: (@nillist, ~nillist),
    vec: Vec<@nillist>,
    res: r
}

struct r {
  _l: @nillist,
}

#[unsafe_destructor]
impl Drop for r {
    fn drop(&mut self) {}
}

fn r(l: @nillist) -> r {
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
                managed: @Nil,
                unique: ~Nil,
                tuple: (@Nil, ~Nil),
                vec: vec!(@Nil),
                res: r(@Nil)
            }
          }
          Some(st) => {
            State {
                managed: @Cons((), st.managed),
                unique: ~Cons((), @*st.unique),
                tuple: (@Cons((), st.tuple.ref0().clone()),
                        ~Cons((), @*st.tuple.ref1().clone())),
                vec: st.vec.clone().append(&[@Cons((), *st.vec.last().unwrap())]),
                res: r(@Cons((), st.res._l))
            }
          }
        };

        recurse_or_fail(depth, Some(st));
    }
}
