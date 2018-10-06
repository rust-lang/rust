// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![allow(unreachable_patterns)]
#![allow(unused_variables)]
#![warn(unused_parens)]

struct A {
    field: Option<String>,
}

fn main() {
    let x = 3;
    match x {
        (_) => {}     //~ WARNING: unnecessary parentheses around pattern
        (y) => {}     //~ WARNING: unnecessary parentheses around pattern
        (ref r) => {} //~ WARNING: unnecessary parentheses around pattern
        e @ 1...2 | (e @ (3...4)) => {}
        //~^ WARNING: unnecessary parentheses around pattern (3 ... 4)
        //~^ WARNING: unnecessary parentheses around pattern (e @ _)
    }

    let field = "foo".to_string();
    let x: Option<A> = Some(A { field: Some(field) });
    match x {
        Some(A {
            field: (ref a @ Some(_)),
            //~^ WARNING: unnecessary parentheses around pattern
            ..
        }) => {}
        _ => {}
    }
}
