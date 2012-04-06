// -*- rust -*-
// Tests that a function with a ! annotation always actually fails

fn bad_bang(i: uint) -> ! {
    ret 7u;
    //!^ ERROR expected `_|_` but found `uint`
}

fn main() { bad_bang(5u); }
