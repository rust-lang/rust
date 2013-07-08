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

// GDB doesn't know about UTF-32 character encoding and will print a rust char as only its numerical
// value.

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run
// debugger:finish

// debugger:print a
// check:$1 = 9898

// debugger:print b
// check:$2 = false

fn main() {
    let (a, b) : (int, bool) = (9898, false);

    zzz();
}

fn zzz() {()}