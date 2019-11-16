#![feature(slice_patterns)]
#![deny(unreachable_patterns)]

fn main() {
    let s: &[bool] = &[true; 0];

    match s {
        [true, ..] => {}
        [true, ..] => {} //~ ERROR unreachable pattern
        [true] => {} //~ ERROR unreachable pattern
        [..] => {}
    }
    match s {
        [.., true] => {}
        [.., true] => {} //~ ERROR unreachable pattern
        [true] => {} //~ ERROR unreachable pattern
        [..] => {}
    }
    match s {
        [false, .., true] => {}
        [false, .., true] => {} //~ ERROR unreachable pattern
        [false, true] => {} //~ ERROR unreachable pattern
        [false] => {}
        [..] => {}
    }
}
