// FIXME(arora-aman) add run-pass once 2229 is implemented

#![feature(capture_disjoint_fields)]
//~^ warning the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

struct Point {
    x: i32,
    y: i32,
}

// This testcase ensures that nested closures are handles properly
// - The nested closure is analyzed first.
// - The capture kind of the nested closure is accounted for by the enclosing closure
// - Any captured path by the nested closure that starts off a local variable in the enclosing
// closure is not listed as a capture of the enclosing closure.

fn main() {
    let mut p = Point { x: 5, y: 20 };

    let mut c1 = #[rustc_capture_analysis]
        //~^ ERROR: attributes on expressions are experimental
    || {
        println!("{}", p.x);
        //~^ ERROR: Capturing p[(0, 0)] -> ImmBorrow
        //~^^ ERROR: Min Capture p[(0, 0)] -> ImmBorrow
        let incr = 10;
        let mut c2 = #[rustc_capture_analysis]
        //~^ ERROR: attributes on expressions are experimental
        || p.y += incr;
        //~^ ERROR: Capturing p[(1, 0)] -> MutBorrow
        //~^^ ERROR: Capturing incr[] -> ImmBorrow
        //~^^^ ERROR: Min Capture p[(1, 0)] -> MutBorrow
        //~^^^^ ERROR: Min Capture incr[] -> ImmBorrow
        //~^^^^^ ERROR: Capturing p[(1, 0)] -> MutBorrow
        //~^^^^^^ ERROR: Min Capture p[(1, 0)] -> MutBorrow
        c2();
        println!("{}", p.y);
    };

    c1();

    let px = &p.x;

    println!("{}", px);

    c1();
}
