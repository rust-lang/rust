

// -*- rust -*-
use std;
import std::str;

fn main() {
    let a: istr = ~"this \
is a test";
    let b: istr =
        ~"this \
               is \
               another \
               test";
    assert (str::eq(a, ~"this is a test"));
    assert (str::eq(b, ~"this is another test"));
}
