// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(path_statement)]
#![allow(unreachable_code)]
#![allow(unused_variable)]

fn id(x: bool) -> bool { x }

fn call_id() {
    let c = fail!();
    id(c);
}

fn call_id_2() { id(true) && id(return); }

fn call_id_3() { id(return) && id(return); }

fn ret_ret() -> int { return (return 2) + 3; }

fn ret_guard() {
    match 2 {
      x if (return) => { x; }
      _ => {}
    }
}

pub fn main() {}
