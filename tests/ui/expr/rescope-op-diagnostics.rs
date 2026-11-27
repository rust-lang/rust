//! Test diagnostics for malformed invocations of `scope!` and `extend!`

#![feature(temporary_rescoping)]

use std::temporary_rescoping::{scope, extend};

fn main() {
    scope!();  //~ ERROR: expected a label
    extend!(); //~ ERROR: expected a label
    scope!(a => ());  //~ ERROR: expected a label
    extend!(a => ()); //~ ERROR: expected a label
    scope!('a => ());  //~ ERROR: FIXME: rescoping operators are not implemented
    extend!('a => ()); //~ ERROR: FIXME: rescoping operators are not implemented
    'a: {
        scope!('a);  //~ ERROR: expected `=>`, found `)`
        extend!('a); //~ ERROR: expected `=>`, found `)`
        scope!('a =>);  //~ ERROR: expected expression, found `)`
        extend!('a =>); //~ ERROR: expected expression, found `)`
    }
}
