// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    // bad arguments to the format! call

    format!("{}");            //~ ERROR: invalid reference to argument

    format!("{1}", 1);        //~ ERROR: invalid reference to argument `1`
                            //~^ ERROR: argument never used
    format!("{foo}");         //~ ERROR: no argument named `foo`

    format!("{}", 1, 2);               //~ ERROR: argument never used
    format!("{1}", 1, 2);              //~ ERROR: argument never used
    format!("{}", 1, foo=2);           //~ ERROR: named argument never used
    format!("{foo}", 1, foo=2);        //~ ERROR: argument never used
    format!("", foo=2);                //~ ERROR: named argument never used

    format!("{0:d} {0:s}", 1);         //~ ERROR: redeclared with type `s`
    format!("{foo:d} {foo:s}", foo=1); //~ ERROR: redeclared with type `s`

    format!("{foo}", foo=1, foo=2);    //~ ERROR: duplicate argument
    format!("", foo=1, 2);             //~ ERROR: positional arguments cannot follow

    // bad syntax of the format string

    format!("{"); //~ ERROR: expected `}` but string was terminated

    format!("foo } bar"); //~ ERROR: unmatched `}` found
    format!("foo }"); //~ ERROR: unmatched `}` found

    format!();          //~ ERROR: requires at least a format string argument
    format!("" 1);      //~ ERROR: expected token: `,`
    format!("", 1 1);   //~ ERROR: expected token: `,`
}
