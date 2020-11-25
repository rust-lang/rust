// run-pass

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut p = Point { x: 5, y: 20 };

    let mut c1 = || {
        println!("{}", p.x);

        let incr = 10;

        let mut c2 = || p.y += incr;
        c2();

        println!("{}", p.y);
    };

    c1();

    let px = &p.x;

    println!("{}", px);

    c1();
}
