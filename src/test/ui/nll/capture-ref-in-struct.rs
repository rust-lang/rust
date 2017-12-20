// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Znll-dump-cause

// Test that a structure which tries to store a pointer to `y` into
// `p` (indirectly) fails to compile.

#![feature(rustc_attrs)]
#![feature(nll)]

struct SomeStruct<'a, 'b: 'a> {
    p: &'a mut &'b i32,
    y: &'b i32,
}

fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;

        let closure = SomeStruct {
            p: &mut p,
            y: &y,
            //~^ ERROR `y` does not live long enough [E0597]
        };

        closure.invoke();
    }

    deref(p);
}

impl<'a, 'b> SomeStruct<'a, 'b> {
    fn invoke(self) {
        *self.p = self.y;
    }
}

fn deref(_: &i32) { }

fn main() { }
