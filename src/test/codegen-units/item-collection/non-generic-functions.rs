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
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]

//~ TRANS_ITEM fn non_generic_functions::foo[0]
fn foo() {
    {
        //~ TRANS_ITEM fn non_generic_functions::foo[0]::foo[0]
        fn foo() {}
        foo();
    }

    {
        //~ TRANS_ITEM fn non_generic_functions::foo[0]::foo[1]
        fn foo() {}
        foo();
    }
}

//~ TRANS_ITEM fn non_generic_functions::bar[0]
fn bar() {
    //~ TRANS_ITEM fn non_generic_functions::bar[0]::baz[0]
    fn baz() {}
    baz();
}

struct Struct { _x: i32 }

impl Struct {
    //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]
    fn foo() {
        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]::foo[0]
            fn foo() {}
            foo();
        }

        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::foo[0]::foo[1]
            fn foo() {}
            foo();
        }
    }

    //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]
    fn bar(&self) {
        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]::foo[0]
            fn foo() {}
            foo();
        }

        {
            //~ TRANS_ITEM fn non_generic_functions::{{impl}}[0]::bar[0]::foo[1]
            fn foo() {}
            foo();
        }
    }
}

//~ TRANS_ITEM fn non_generic_functions::main[0]
fn main() {
    foo();
    bar();
    Struct::foo();
    let x = Struct { _x: 0 };
    x.bar();
}

//~ TRANS_ITEM drop-glue i8
