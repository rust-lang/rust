

// -*- rust -*-
use std;
import std::istr;

fn main() {
    let a: istr = ~"this \
is a test";
    let b: istr =
        ~"this \
               is \
               another \
               test";
    assert (istr::eq(a, ~"this is a test"));
    assert (istr::eq(b, ~"this is another test"));
}
