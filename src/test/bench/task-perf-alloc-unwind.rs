// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax, time2, vec_push_all)]

use std::env;
use std::thread;
use std::time::Instant;

#[derive(Clone)]
enum List<T> {
    Nil, Cons(T, Box<List<T>>)
}

fn main() {
    let (repeat, depth) = if env::var_os("RUST_BENCH").is_some() {
        (50, 1000)
    } else {
        (10, 10)
    };

    run(repeat, depth);
}

fn run(repeat: isize, depth: isize) {
    for _ in 0..repeat {
        let start = Instant::now();
        let _ = thread::spawn(move|| {
            recurse_or_panic(depth, None)
        }).join();
        println!("iter: {:?}", start.elapsed());
    }
}

type nillist = List<()>;

// Filled with things that have to be unwound

struct State {
    unique: Box<nillist>,
    vec: Vec<Box<nillist>>,
    res: r
}

struct r {
  _l: Box<nillist>,
}

impl Drop for r {
    fn drop(&mut self) {}
}

fn r(l: Box<nillist>) -> r {
    r {
        _l: l
    }
}

fn recurse_or_panic(depth: isize, st: Option<State>) {
    if depth == 0 {
        panic!();
    } else {
        let depth = depth - 1;

        let st = match st {
            None => {
                State {
                    unique: box List::Nil,
                    vec: vec!(box List::Nil),
                    res: r(box List::Nil)
                }
            }
            Some(st) => {
                let mut v = st.vec.clone();
                v.push_all(&[box List::Cons((), st.vec.last().unwrap().clone())]);
                State {
                    unique: box List::Cons((), box *st.unique),
                    vec: v,
                    res: r(box List::Cons((), st.res._l.clone())),
                }
            }
        };

        recurse_or_panic(depth, Some(st));
    }
}
