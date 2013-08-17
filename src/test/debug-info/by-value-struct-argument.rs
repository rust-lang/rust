// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Does not work yet, see issue #8512
// xfail-test

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run

// debugger:finish
// debugger:print s
// check:$1 = {a = 1, b = 2.5}
// debugger:continue

#[deriving(Clone)]
struct Struct {
    a: int,
    b: float
}

fn fun(s: Struct) {
    zzz();
}

fn main() {
    fun(Struct { a: 1, b: 2.5 });
}

fn zzz() {()}
