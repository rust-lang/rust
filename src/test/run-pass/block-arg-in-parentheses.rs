// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f(_: proc(int, int) -> int) -> int {
    10
}

fn w_semi() {
    // the semicolon causes compiler not to
    // complain about the ignored return value:
    do f |x, y| { x+y };
}

fn w_paren1() -> int {
    (do f |x, y| { x+y }) - 10
}

fn w_paren2() -> int {
    (do f |x, y| { x+y } - 10)
}

fn w_ret() -> int {
    return do f |x, y| { x+y } - 10;
}

pub fn main() {
    w_semi();
    w_paren1();
    w_paren2();
    w_ret();
}
