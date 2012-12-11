// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn id(x: bool) -> bool { x }

fn call_id() {
    let c = move fail;
    id(c);
}

fn call_id_2() { id(true) && id(return); }

fn call_id_3() { id(return) && id(return); }

fn log_fail() { log(error, fail); }

fn log_ret() { log(error, return); }

fn log_break() { loop { log(error, break); } }

fn log_again() { loop { log(error, loop); } }

fn ret_ret() -> int { return (return 2) + 3; }

fn ret_guard() {
    match 2 {
      x if (return) => { x; }
      _ => {}
    }
}

fn main() {}
