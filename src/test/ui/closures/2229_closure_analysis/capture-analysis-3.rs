#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

#[derive(Debug)]
struct Child {
    c: String,
    d: String,
}

#[derive(Debug)]
struct Parent {
    b: Child,
}

fn main() {
    let mut a = Parent { b: Child {c: String::new(), d: String::new()} };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
    //~| Min Capture analysis includes:
        let _x = a.b.c;
        //~^ NOTE: Capturing a[(0, 0),(0, 0)] -> ByValue
        //~| NOTE: a[(0, 0)] captured as ByValue here
        println!("{:?}", a.b);
        //~^ NOTE: Capturing a[(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture a[(0, 0)] -> ByValue
        //~| NOTE: a[(0, 0)] used here
    };
}
