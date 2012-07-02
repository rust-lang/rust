// -*- rust -*-
// Tests that a function with a ! annotation always actually fails

fn bad_bang(i: uint) -> ! {
    if i < 0u { } else { fail; }
    //~^ ERROR expected `_|_` but found `()`
}

fn main() { bad_bang(5u); }
