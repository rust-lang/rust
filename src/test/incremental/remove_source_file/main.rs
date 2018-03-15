// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test case makes sure that the compiler doesn't crash due to a failing
// table lookup when a source file is removed.

// revisions:rpass1 rpass2

// Note that we specify -g so that the FileMaps actually get referenced by the
// incr. comp. cache:
// compile-flags: -Z query-dep-graph -g

#[cfg(rpass1)]
mod auxiliary;

#[cfg(rpass1)]
fn main() {
    auxiliary::print_hello();
}

#[cfg(rpass2)]
fn main() {
    println!("hello");
}
