// run-pass

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

// Tests that if a closure uses indivual fields of the same object
// then that case is handled properly.

#![allow(unused)]

struct Struct {
    x: i32,
    y: i32,
    s: String,
}

fn main() {
    let mut s = Struct { x: 10, y: 10, s: String::new() };

    let mut c = {
        s.x += 10;
        s.y += 42;
        s.s = String::from("new");
    };
}
