#![warn(clippy::all)]

use std::cmp::max as my_max;
use std::cmp::min as my_min;
use std::cmp::{max, min};

const LARGE: usize = 3;

fn main() {
    let x;
    x = 2usize;
    min(1, max(3, x));
    min(max(3, x), 1);
    max(min(x, 1), 3);
    max(3, min(x, 1));

    my_max(3, my_min(x, 1));

    min(3, max(1, x)); // ok, could be 1, 2 or 3 depending on x

    min(1, max(LARGE, x)); // no error, we don't lookup consts here

    let y = 2isize;
    min(max(y, -1), 3);

    let s;
    s = "Hello";

    min("Apple", max("Zoo", s));
    max(min(s, "Apple"), "Zoo");

    max("Apple", min(s, "Zoo")); // ok
}
