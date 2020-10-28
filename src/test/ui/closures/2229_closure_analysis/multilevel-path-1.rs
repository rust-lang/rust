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

    // Only paths that appears within the closure that directly start off
    // a variable defined outside the closure are captured.
    //
    // Therefore `w.p` is captured
    // Note that `wp.x` doesn't start off a variable defined outside the closure.
    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        let wp = &w.p;
        //~^ ERROR: Capturing w[(0, 0)] -> ImmBorrow
        //~^^ ERROR: Min Capture w[(0, 0)] -> ImmBorrow
        println!("{}", wp.x);
    };

    // Since `c` captures `w.p` by an ImmBorrow, `w.p.y` can't be mutated.
    let py = &mut w.p.y;
    c();

    *py = 20
}
