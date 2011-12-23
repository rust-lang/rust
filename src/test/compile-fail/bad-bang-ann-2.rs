// -*- rust -*-
// Tests that a function with a ! annotation always actually fails
// error-pattern: some control paths may return

fn bad_bang(i: uint) -> ! { log(debug, 3); }

fn main() { bad_bang(5u); }
