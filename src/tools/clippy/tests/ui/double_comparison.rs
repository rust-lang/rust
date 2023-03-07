// run-rustfix

fn main() {
    let x = 1;
    let y = 2;
    if x == y || x < y {
        // do something
    }
    if x < y || x == y {
        // do something
    }
    if x == y || x > y {
        // do something
    }
    if x > y || x == y {
        // do something
    }
    if x < y || x > y {
        // do something
    }
    if x > y || x < y {
        // do something
    }
    if x <= y && x >= y {
        // do something
    }
    if x >= y && x <= y {
        // do something
    }
}
