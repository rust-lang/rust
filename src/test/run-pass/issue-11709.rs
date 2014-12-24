// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty

// Don't panic on blocks without results
// There are several tests in this run-pass that raised
// when this bug was opened. The cases where the compiler
// panics before the fix have a comment.

#![feature(default_type_params)]

use std::thunk::Thunk;

struct S {x:()}

fn test(slot: &mut Option<Thunk<(),Thunk>>) -> () {
  let a = slot.take();
  let _a = match a {
    // `{let .. a(); }` would break
    Some(a) => { let _a = a.invoke(()); },
    None => (),
  };
}

fn not(b: bool) -> bool {
    if b {
        !b
    } else {
        // `panic!(...)` would break
        panic!("Break the compiler");
    }
}

pub fn main() {
    // {} would break
    let _r = {};
    let mut slot = None;
    // `{ test(...); }` would break
    let _s : S  = S{ x: { test(&mut slot); } };

    let _b = not(true);
}
