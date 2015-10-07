// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:stop

// #18576
// Make sure that calling an extern function pointer in an unreachable
// context doesn't cause an LLVM assertion

#[allow(unreachable_code)]
fn main() {
    panic!("stop");
    let pointer = other;
    pointer();
}
extern fn other() {}
