#![feature(slice_patterns)]

fn main() {
    let s: [bool; 1] = [false; 1];
    match s {
        [a] => {}
    }
    match s {
        [a, ..] => {}
    }
    match s {
        [true, ..] => {}
        [.., false] => {}
    }

    let s: [bool; 2] = [false; 2];
    match s {
        [a, b] => {}
    }
    match s {
        [a, ..] => {}
    }
    match s {
    //~^ ERROR `[false, true]` not covered
        [true, ..] => {}
        [.., false] => {}
    }
}
