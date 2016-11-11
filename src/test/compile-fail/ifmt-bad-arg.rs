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

    format!("", 1, 2);                 //~ ERROR: multiple unused formatting arguments
    format!("{}", 1, 2);               //~ ERROR: argument never used
    format!("{1}", 1, 2);              //~ ERROR: argument never used
    format!("{}", 1, foo=2);           //~ ERROR: named argument never used
    format!("{foo}", 1, foo=2);        //~ ERROR: argument never used
    format!("", foo=2);                //~ ERROR: named argument never used

    format!("{foo}", foo=1, foo=2);    //~ ERROR: duplicate argument
    format!("", foo=1, 2);             //~ ERROR: positional arguments cannot follow

    // bad number of arguments, see #15780

    format!("{0}");
    //~^ ERROR invalid reference to argument `0` (no arguments given)

    format!("{0} {1}", 1);
    //~^ ERROR invalid reference to argument `1` (there is 1 argument)

    format!("{0} {1} {2}", 1, 2);
    //~^ ERROR invalid reference to argument `2` (there are 2 arguments)

    format!("{0} {1}");
    //~^ ERROR invalid reference to argument `0` (no arguments given)
    //~^^ ERROR invalid reference to argument `1` (no arguments given)

    // bad named arguments, #35082

    format!("{valuea} {valueb}", valuea=5, valuec=7);
    //~^ ERROR there is no argument named `valueb`
    //~^^ ERROR named argument never used

    // bad syntax of the format string

    format!("{"); //~ ERROR: expected `'}'` but string was terminated

    format!("foo } bar"); //~ ERROR: unmatched `}` found
    format!("foo }"); //~ ERROR: unmatched `}` found

    format!("foo %s baz", "bar"); //~ ERROR: argument never used
}
