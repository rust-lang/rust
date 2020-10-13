// FIXME(arora-aman) add run-pass once 2229 is implemented

#![feature(capture_disjoint_fields)]
//~^ warning the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]
#![allow(unused)]

struct Point {
    x: i32,
    y: i32,
}
struct Wrapper {
    p: Point,
}

fn main() {
    let mut w = Wrapper { p: Point { x: 10, y: 10 } };

    let c = #[rustc_capture_analysis]
        //~^ ERROR: attributes on expressions are experimental
    || {
        println!("{}", w.p.x);
    };

    // `c` only captures `w.p.x`, therefore it's safe to mutate `w.p.y`.
    let py = &mut w.p.y;
    c();

    *py = 20
}
