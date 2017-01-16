// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-const_fn

// Test use of const fn without feature gate.

const fn foo() -> usize { 0 } //~ ERROR const fn is unstable

trait Foo {
    const fn foo() -> u32; //~ ERROR const fn is unstable
                           //~| ERROR trait fns cannot be declared const
    const fn bar() -> u32 { 0 } //~ ERROR const fn is unstable
                                //~| ERROR trait fns cannot be declared const
}

impl Foo {
    const fn baz() -> u32 { 0 } //~ ERROR const fn is unstable
}

impl Foo for u32 {
    const fn foo() -> u32 { 0 } //~ ERROR const fn is unstable
                                //~| ERROR trait fns cannot be declared const
}

static FOO: usize = foo();
const BAR: usize = foo();

macro_rules! constant {
    ($n:ident: $t:ty = $v:expr) => {
        const $n: $t = $v;
    }
}

constant! {
    BAZ: usize = foo()
}

fn main() {
    let x: [usize; foo()] = [];
}
