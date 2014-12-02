// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)
// min-lldb-version: 310

// compile-flags:-g

#![feature(non_ascii_idents)]

// This test checks whether debuginfo generation can handle multi-byte UTF-8
// characters at the end of a block. There's no need to do anything in the
// debugger -- just make sure that the compiler doesn't crash.
// See also issue #18791.

struct C { θ: u8 }

fn main() {
    let x =  C { θ: 0 };
    (|c: C| c.θ )(x);
}
