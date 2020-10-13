#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

fn main() {
    let mut t = (10, 10);

    let c = #[rustc_capture_analysis]
    || {
        println!("{}", t.0);
    };

    // `c` only captures t.0, therefore mutating t.1 is allowed.
    let t1 = &mut t.1;

    c();
    *t1 = 20;
}
