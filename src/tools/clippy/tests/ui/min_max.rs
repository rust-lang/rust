#![allow(clippy::manual_clamp)]

use std::cmp::{max as my_max, max, min as my_min, min};

const LARGE: usize = 3;

struct NotOrd(u64);

impl NotOrd {
    fn min(self, x: u64) -> NotOrd {
        NotOrd(x)
    }

    fn max(self, x: u64) -> NotOrd {
        NotOrd(x)
    }
}

fn main() {
    let x = 2usize;
    min(1, max(3, x));
    //~^ min_max

    min(max(3, x), 1);
    //~^ min_max

    max(min(x, 1), 3);
    //~^ min_max

    max(3, min(x, 1));
    //~^ min_max

    my_max(3, my_min(x, 1));
    //~^ min_max

    min(3, max(1, x)); // ok, could be 1, 2 or 3 depending on x

    min(1, max(LARGE, x)); // no error, we don't lookup consts here

    let y = 2isize;
    min(max(y, -1), 3);

    let s = "Hello";
    min("Apple", max("Zoo", s));
    //~^ min_max

    max(min(s, "Apple"), "Zoo");
    //~^ min_max

    max("Apple", min(s, "Zoo")); // ok

    let f = 3f32;
    x.min(1).max(3);
    //~^ min_max

    x.max(3).min(1);
    //~^ min_max

    f.max(3f32).min(1f32);
    //~^ min_max

    x.max(1).min(3); // ok
    x.min(3).max(1); // ok
    f.min(3f32).max(1f32); // ok

    max(x.min(1), 3);
    //~^ min_max

    min(x.max(1), 3); // ok

    s.max("Zoo").min("Apple");
    //~^ min_max

    s.min("Apple").max("Zoo");
    //~^ min_max

    s.min("Zoo").max("Apple"); // ok

    let not_ord = NotOrd(1);
    not_ord.min(1).max(3); // ok
}
