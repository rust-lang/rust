// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

//~ MONO_ITEM fn non_generic_functions::foo[0]
fn foo() {
    {
        //~ MONO_ITEM fn non_generic_functions::foo[0]::foo[0]
        fn foo() {}
        foo();
    }

    {
        //~ MONO_ITEM fn non_generic_functions::foo[0]::foo[1]
        fn foo() {}
        foo();
    }
}

//~ MONO_ITEM fn non_generic_functions::bar[0]
fn bar() {
    //~ MONO_ITEM fn non_generic_functions::bar[0]::baz[0]
    fn baz() {}
    baz();
}

struct Struct { _x: i32 }

impl Struct {
    //~ MONO_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]
    fn foo() {
        {
            //~ MONO_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]::foo[0]
            fn foo() {}
            foo();
        }

        {
            //~ MONO_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]::foo[1]
            fn foo() {}
            foo();
        }
    }

    //~ MONO_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]
    fn bar(&self) {
        {
            //~ MONO_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]::foo[0]
            fn foo() {}
            foo();
        }

        {
            //~ MONO_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]::foo[1]
            fn foo() {}
            foo();
        }
    }
}

//~ MONO_ITEM fn non_generic_functions::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    foo();
    bar();
    Struct::foo();
    let x = Struct { _x: 0 };
    x.bar();

    0
}
