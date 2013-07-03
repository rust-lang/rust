// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// FIXME: #7336: codegen bug makes this segfault on Linux x86_64

fn foo<'a>(blk: &fn(p: &'a fn() -> &'a fn())) {
    let mut state = 0;
    let statep = &mut state;
    do blk {
        || { *statep = 1; }
    }
}
fn main() {
    do foo |p| { p()() }
}
