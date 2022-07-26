// run-pass
#![allow(unreachable_code)]
#![allow(for_loop_over_fallibles)]
#![deny(unused_variables)]

fn main() {
    for _ in match return () {
        () => Some(0),
    } {}
}
