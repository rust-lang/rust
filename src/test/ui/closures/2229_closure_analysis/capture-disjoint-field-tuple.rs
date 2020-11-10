// FIXME(arora-aman) add run-pass once 2229 is implemented

#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

fn main() {
    let mut t = (10, 10);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        println!("{}", t.0);
        //~^ ERROR: Capturing t[(0, 0)] -> ImmBorrow
        //~| ERROR: Min Capture t[(0, 0)] -> ImmBorrow
    };

    // `c` only captures t.0, therefore mutating t.1 is allowed.
    let t1 = &mut t.1;

    c();
    *t1 = 20;
}
