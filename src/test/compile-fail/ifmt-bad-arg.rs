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
    format!("#");                      //~ ERROR: `#` reference used
    format!("", foo=1, 2);             //~ ERROR: positional arguments cannot follow

    format!("{0, select, a{} a{} other{}}", "a");    //~ ERROR: duplicate selector
    format!("{0, plural, =1{} =1{} other{}}", 1u);   //~ ERROR: duplicate selector
    format!("{0, plural, one{} one{} other{}}", 1u); //~ ERROR: duplicate selector

    // bad syntax of the format string

    format!("{"); //~ ERROR: unterminated format string
    format!("\\ "); //~ ERROR: invalid escape
    format!("\\"); //~ ERROR: expected an escape

    format!("{0, }", 1); //~ ERROR: expected method
    format!("{0, foo}", 1); //~ ERROR: unknown method
    format!("{0, select}", "a"); //~ ERROR: must be followed by
    format!("{0, plural}", 1); //~ ERROR: must be followed by

    format!("{0, select, a{{}", 1); //~ ERROR: must be terminated
    format!("{0, select, {} other{}}", "a"); //~ ERROR: empty selector
    format!("{0, select, other{} other{}}", "a"); //~ ERROR: multiple `other`
    format!("{0, plural, offset: other{}}", "a"); //~ ERROR: must be an integer
    format!("{0, plural, offset 1 other{}}", "a"); //~ ERROR: be followed by `:`
    format!("{0, plural, =a{} other{}}", "a"); //~ ERROR: followed by an integer
    format!("{0, plural, a{} other{}}", "a"); //~ ERROR: unexpected plural
    format!("{0, select, a{}}", "a"); //~ ERROR: must provide an `other`
    format!("{0, plural, =1{}}", "a"); //~ ERROR: must provide an `other`

    format!("{0, plural, other{{0:s}}}", "a"); //~ ERROR: previously used as
    format!("{:s} {0, plural, other{}}", "a"); //~ ERROR: argument used to
    format!("{0, select, other{}} \
             {0, plural, other{}}", "a");
    //~^ ERROR: declared with multiple formats

    // It should be illegal to use implicit placement arguments nested inside of
    // format strings because otherwise the "internal pointer of which argument
    // is next" would be invalidated if different cases had different numbers of
    // arguments.
    format!("{0, select, other{{}}}", "a"); //~ ERROR: cannot use implicit
    format!("{0, plural, other{{}}}", 1); //~ ERROR: cannot use implicit
    format!("{0, plural, other{{1:.*d}}}", 1, 2); //~ ERROR: cannot use implicit

    format!("foo } bar"); //~ ERROR: unmatched `}` found
    format!("foo }"); //~ ERROR: unmatched `}` found

    format!();          //~ ERROR: requires at least a format string argument
    format!("" 1);      //~ ERROR: expected token: `,`
    format!("", 1 1);   //~ ERROR: expected token: `,`
}
