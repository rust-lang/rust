

// -*- rust -*-
use std;
import str;

fn main() {
    let a: ~str = ~"this \
is a test";
    let b: ~str =
        ~"this \
               is \
               another \
               test";
    assert (a == ~"this is a test");
    assert (b == ~"this is another test");
}
