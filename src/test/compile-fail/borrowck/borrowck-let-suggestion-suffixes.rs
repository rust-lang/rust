// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f() {
    let old = ['o'];         // statement 0
    let mut v1 = Vec::new(); // statement 1

    let mut v2 = Vec::new(); // statement 2

    let young = ['y'];       // statement 3

    v2.push(&young[0]);      // statement 4
    //~^ ERROR `young[..]` does not live long enough
    //~| NOTE does not live long enough
    //~| NOTE values in a scope are dropped in the opposite order they are created

    let mut v3 = Vec::new(); // statement 5

    v3.push(&'x');           // statement 6
    //~^ ERROR borrowed value does not live long enough
    //~| NOTE does not live long enough
    //~| NOTE borrowed value only valid until here
    //~| HELP consider using a `let` binding to increase its lifetime

    {

        let mut v4 = Vec::new(); // (sub) statement 0

        v4.push(&'y');
        //~^ ERROR borrowed value does not live long enough
        //~| NOTE does not live long enough
        //~| NOTE borrowed value only valid until here
        //~| HELP consider using a `let` binding to increase its lifetime

    }                       // (statement 7)
    //~^ NOTE borrowed value must be valid until here

    let mut v5 = Vec::new(); // statement 8

    v5.push(&'z');
    //~^ ERROR borrowed value does not live long enough
    //~| NOTE does not live long enough
    //~| NOTE borrowed value only valid until here
    //~| HELP consider using a `let` binding to increase its lifetime

    v1.push(&old[0]);
}
//~^ NOTE borrowed value dropped before borrower
//~| NOTE borrowed value must be valid until here
//~| NOTE borrowed value must be valid until here

fn main() {
    f();
}
