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

    // bad number of arguments, see #44954 (originally #15780)

    format!("{}");
    //~^ ERROR: 1 positional argument in format string, but no arguments were given

    format!("{1}", 1);
    //~^ ERROR: invalid reference to positional argument 1 (there is 1 argument)
    //~^^ ERROR: argument never used

    format!("{} {}");
    //~^ ERROR: 2 positional arguments in format string, but no arguments were given

    format!("{0} {1}", 1);
    //~^ ERROR: invalid reference to positional argument 1 (there is 1 argument)

    format!("{0} {1} {2}", 1, 2);
    //~^ ERROR: invalid reference to positional argument 2 (there are 2 arguments)

    format!("{} {value} {} {}", 1, value=2);
    //~^ ERROR: invalid reference to positional argument 2 (there are 2 arguments)
    format!("{name} {value} {} {} {} {} {} {}", 0, name=1, value=2);
    //~^ ERROR: invalid reference to positional arguments 3, 4 and 5 (there are 3 arguments)

    format!("{} {foo} {} {bar} {}", 1, 2, 3);
    //~^ ERROR: there is no argument named `foo`
    //~^^ ERROR: there is no argument named `bar`

    format!("{foo}");                //~ ERROR: no argument named `foo`
    format!("", 1, 2);               //~ ERROR: multiple unused formatting arguments
    format!("{}", 1, 2);             //~ ERROR: argument never used
    format!("{1}", 1, 2);            //~ ERROR: argument never used
    format!("{}", 1, foo=2);         //~ ERROR: named argument never used
    format!("{foo}", 1, foo=2);      //~ ERROR: argument never used
    format!("", foo=2);              //~ ERROR: named argument never used
    format!("{} {}", 1, 2, foo=1, bar=2);  //~ ERROR: multiple unused formatting arguments

    format!("{foo}", foo=1, foo=2);  //~ ERROR: duplicate argument
    format!("", foo=1, 2);           //~ ERROR: positional arguments cannot follow

    // bad named arguments, #35082

    format!("{valuea} {valueb}", valuea=5, valuec=7);
    //~^ ERROR there is no argument named `valueb`
    //~^^ ERROR named argument never used

    // bad syntax of the format string

    format!("{"); //~ ERROR: expected `'}'` but string was terminated

    format!("foo } bar"); //~ ERROR: unmatched `}` found
    format!("foo }"); //~ ERROR: unmatched `}` found

    format!("foo %s baz", "bar"); //~ ERROR: argument never used

    format!(r##"

        {foo}

    "##);
    //~^^^ ERROR: there is no argument named `foo`
}
