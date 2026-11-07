#![feature(half_open_range_patterns_in_slices)]

fn main() {
    let xs = [13, 1, 5, 2, 3, 1, 21, 8];
    let [a, b, c, rest @ ..] = xs;
    // Consider the following example:
    assert!(a == 13 && b == 1 && c == 5 && rest.len() == 5);

    // What if we wanted to pull this apart without individually binding a, b, and c?
    let [first_three @ ..3, rest @ 2..] = xs;
    //~^ ERROR pattern requires 2 elements but array has 8
    // This is somewhat unintuitive and makes slice patterns exceedingly verbose.
    // We want to stabilize half-open RangeFrom (`X..`) patterns
    // but without banning us from using them for a more efficient slice pattern syntax.
}
