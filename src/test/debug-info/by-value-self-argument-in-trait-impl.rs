// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android: FIXME(#10381)

#[feature(managed_boxes)];

// compile-flags:-g
// debugger:rbreak zzz
// debugger:run

// debugger:finish
// debugger:print self
// check:$1 = 1111
// debugger:continue

// debugger:finish
// debugger:print self
// check:$2 = {x = 2222, y = 3333}
// debugger:continue

// debugger:finish
// debugger:print self
// check:$3 = {4444.5, 5555, 6666, 7777.5}
// debugger:continue

// debugger:finish
// debugger:print self->val
// check:$4 = 8888
// debugger:continue

trait Trait {
    fn method(self) -> Self;
}

impl Trait for int {
    fn method(self) -> int {
        zzz();
        self
    }
}

struct Struct {
    x: uint,
    y: uint,
}

impl Trait for Struct {
    fn method(self) -> Struct {
        zzz();
        self
    }
}

impl Trait for (f64, int, int, f64) {
    fn method(self) -> (f64, int, int, f64) {
        zzz();
        self
    }
}

impl Trait for @int {
    fn method(self) -> @int {
        zzz();
        self
    }
}

fn main() {
    let _ = (1111 as int).method();
    let _ = Struct { x: 2222, y: 3333 }.method();
    let _ = (4444.5, 5555, 6666, 7777.5).method();
    let _ = (@8888).method();
}

fn zzz() {()}
