// edition:2018

// Enabling `ireffutable_let_patterns` isn't necessary for what this tests, but it makes coming up
// with examples easier.
#![feature(irrefutable_let_patterns)]

#[allow(irrefutable_let_patterns)]
fn main() {
    use std::ops::Range;

    if let Range { start: _, end: _ } = true..true && false { }
    //~^ ERROR ambiguous use of `&&`

    if let Range { start: _, end: _ } = true..true || false { }
    //~^ ERROR ambiguous use of `||`

    while let Range { start: _, end: _ } = true..true && false { }
    //~^ ERROR ambiguous use of `&&`

    while let Range { start: _, end: _ } = true..true || false { }
    //~^ ERROR ambiguous use of `||`

    if let true = false && false { }
    //~^ ERROR ambiguous use of `&&`

    while let true = (1 == 2) && false { }
    //~^ ERROR ambiguous use of `&&`

    // The following cases are not an error as parenthesis are used to
    // clarify intent:

    if let Range { start: _, end: _ } = true..(true || false) { }

    if let Range { start: _, end: _ } = true..(true && false) { }

    while let Range { start: _, end: _ } = true..(true || false) { }

    while let Range { start: _, end: _ } = true..(true && false) { }
}
