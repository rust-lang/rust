#![allow(clippy::needless_ifs)]

fn main() {
    let x = 1;
    let y = 2;
    if x == y || x < y {
        //~^ double_comparisons
        // do something
    }
    if x < y || x == y {
        //~^ double_comparisons
        // do something
    }
    if x == y || x > y {
        //~^ double_comparisons
        // do something
    }
    if x > y || x == y {
        //~^ double_comparisons
        // do something
    }
    if x < y || x > y {
        //~^ double_comparisons
        // do something
    }
    if x > y || x < y {
        //~^ double_comparisons
        // do something
    }
    if x <= y && x >= y {
        //~^ double_comparisons
        // do something
    }
    if x >= y && x <= y {
        //~^ double_comparisons
        // do something
    }
}
