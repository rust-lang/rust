// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage0: new feature, remove this when SNAP
// revisions: a b

#[cfg(a)]
mod a {
    const FOO: fn(u8) -> u8 = |v: u8| { v };
    //[a]~^ ERROR non-capturing closure to fn coercion is experimental
    //[a]~^^ ERROR mismatched types

    const BAR: [fn(&mut u32); 1] = [
        |v: &mut u32| *v += 1,
    //[a]~^ ERROR non-capturing closure to fn coercion is experimental
    //[a]~^^ ERROR mismatched types
    ];
}

#[cfg(b)]
mod b {
    fn func_specific() -> (fn() -> u32) {
        || return 42
        //[b]~^ ERROR non-capturing closure to fn coercion is experimental
        //[b]~^^ ERROR mismatched types
    }
    fn foo() {
        // Items
        assert_eq!(func_specific()(), 42);
        let foo: fn(u8) -> u8 = |v: u8| { v };
        //[b]~^ ERROR non-capturing closure to fn coercion is experimental
        //[b]~^^ ERROR mismatched types
    }

}



