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

// compile-flags:--debuginfo=1

// Make sure functions have proper names
// debugger:info functions
// check:static void [...]main();
// check:static void [...]some_function();
// check:static void [...]some_other_function();
// check:static void [...]zzz();

// debugger:rbreak zzz
// debugger:run

// Make sure there is no information about locals
// debugger:finish
// debugger:info locals
// check:No locals.
// debugger:continue


#[allow(unused_variable)];

struct Struct {
    a: i64,
    b: i32
}

fn main() {
    some_function(101, 202);
}


fn zzz() {()}

fn some_function(a: int, b: int) {
    let some_variable = Struct { a: 11, b: 22 };
    let some_other_variable = 23;
    zzz();
}

fn some_other_function(a: int, b: int) -> bool { true }
