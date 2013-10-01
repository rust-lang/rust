// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use extra::list::{List, Cons, Nil};
use extra::time::precise_time_s;
use std::os;
use std::task;

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
        info2!("starting {:.4f}", precise_time_s());
        do task::try {
            recurse_or_fail(depth, None)
        };
        info2!("stopping {:.4f}", precise_time_s());
    }
}

type nillist = List<()>;

// Filled with things that have to be unwound

struct State {
    box: @nillist,
    unique: ~nillist,
    tuple: (@nillist, ~nillist),
    vec: ~[@nillist],
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
        info2!("unwinding {:.4f}", precise_time_s());
        fail2!();
    } else {
        let depth = depth - 1;

        let st = match st {
          None => {
            State {
                box: @Nil,
                unique: ~Nil,
                tuple: (@Nil, ~Nil),
                vec: ~[@Nil],
                res: r(@Nil)
            }
          }
          Some(st) => {
            State {
                box: @Cons((), st.box),
                unique: ~Cons((), @*st.unique),
                tuple: (@Cons((), st.tuple.first()),
                        ~Cons((), @*st.tuple.second())),
                vec: st.vec + &[@Cons((), *st.vec.last())],
                res: r(@Cons((), st.res._l))
            }
          }
        };

        recurse_or_fail(depth, Some(st));
    }
}
