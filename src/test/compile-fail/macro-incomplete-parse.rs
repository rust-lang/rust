// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! ignored_item {
    () => {
        fn foo() {}
        fn bar() {}
        , //~ ERROR macro expansion ignores token `,`
    }
}

macro_rules! ignored_expr {
    () => ( 1,  //~ ERROR unexpected token: `,`
            2 ) //~ ERROR macro expansion ignores token `2`
}

macro_rules! ignored_pat {
    () => ( 1, 2 ) //~ ERROR macro expansion ignores token `,`
}

ignored_item!(); //~ NOTE caused by the macro expansion here

fn main() {
    ignored_expr!(); //~ NOTE caused by the macro expansion here
    match 1 {
        ignored_pat!() => (), //~ NOTE caused by the macro expansion here
        _ => (),
    }
}
