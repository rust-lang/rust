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
    // bad arguments to the ifmt! call

    ifmt!();                //~ ERROR: expects at least one
    ifmt!("{}");            //~ ERROR: invalid reference to argument

    ifmt!("{1}", 1);        //~ ERROR: invalid reference to argument `1`
                            //~^ ERROR: argument never used
    ifmt!("{foo}");         //~ ERROR: no argument named `foo`

    ifmt!("{}", 1, 2);               //~ ERROR: argument never used
    ifmt!("{1}", 1, 2);              //~ ERROR: argument never used
    ifmt!("{}", 1, foo=2);           //~ ERROR: named argument never used
    ifmt!("{foo}", 1, foo=2);        //~ ERROR: argument never used
    ifmt!("", foo=2);                //~ ERROR: named argument never used

    ifmt!("{0:d} {0:s}", 1);         //~ ERROR: redeclared with type `s`
    ifmt!("{foo:d} {foo:s}", foo=1); //~ ERROR: redeclared with type `s`

    ifmt!("{foo}", foo=1, foo=2);    //~ ERROR: duplicate argument
    ifmt!("#");                      //~ ERROR: `#` reference used
    ifmt!("", foo=1, 2);             //~ ERROR: positional arguments cannot follow
    ifmt!("" 1);                     //~ ERROR: expected token: `,`
    ifmt!("", 1 1);                  //~ ERROR: expected token: `,`

    ifmt!("{0, select, a{} a{} other{}}", "a");    //~ ERROR: duplicate selector
    ifmt!("{0, plural, =1{} =1{} other{}}", 1u);   //~ ERROR: duplicate selector
    ifmt!("{0, plural, one{} one{} other{}}", 1u); //~ ERROR: duplicate selector

    // bad syntax of the format string

    ifmt!("{"); //~ ERROR: unterminated format string
    ifmt!("\\ "); //~ ERROR: invalid escape
    ifmt!("\\"); //~ ERROR: expected an escape

    ifmt!("{0, }", 1); //~ ERROR: expected method
    ifmt!("{0, foo}", 1); //~ ERROR: unknown method
    ifmt!("{0, select}", "a"); //~ ERROR: must be followed by
    ifmt!("{0, plural}", 1); //~ ERROR: must be followed by

    ifmt!("{0, select, a{{}", 1); //~ ERROR: must be terminated
    ifmt!("{0, select, {} other{}}", "a"); //~ ERROR: empty selector
    ifmt!("{0, select, other{} other{}}", "a"); //~ ERROR: multiple `other`
    ifmt!("{0, plural, offset: other{}}", "a"); //~ ERROR: must be an integer
    ifmt!("{0, plural, offset 1 other{}}", "a"); //~ ERROR: be followed by `:`
    ifmt!("{0, plural, =a{} other{}}", "a"); //~ ERROR: followed by an integer
    ifmt!("{0, plural, a{} other{}}", "a"); //~ ERROR: unexpected plural
    ifmt!("{0, select, a{}}", "a"); //~ ERROR: must provide an `other`
    ifmt!("{0, plural, =1{}}", "a"); //~ ERROR: must provide an `other`

    ifmt!("{0, plural, other{{0:s}}}", "a"); //~ ERROR: previously used as
    ifmt!("{:s} {0, plural, other{}}", "a"); //~ ERROR: argument used to
    ifmt!("{0, select, other{}} \
           {0, plural, other{}}", "a");
    //~^ ERROR: declared with multiple formats

    // It should be illegal to use implicit placement arguments nested inside of
    // format strings because otherwise the "internal pointer of which argument
    // is next" would be invalidated if different cases had different numbers of
    // arguments.
    ifmt!("{0, select, other{{}}}", "a"); //~ ERROR: cannot use implicit
    ifmt!("{0, plural, other{{}}}", 1); //~ ERROR: cannot use implicit
    ifmt!("{0, plural, other{{1:.*d}}}", 1, 2); //~ ERROR: cannot use implicit
}
