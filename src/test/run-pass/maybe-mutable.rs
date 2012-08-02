


// -*- rust -*-
fn len(v: ~[const int]) -> uint {
    let mut i = 0u;
    while i < vec::len(v) { i += 1u; }
    return i;
}

fn main() {
    let v0 = ~[1, 2, 3, 4, 5];
    log(debug, len(v0));
    let v1 = ~[mut 1, 2, 3, 4, 5];
    log(debug, len(v1));
}
