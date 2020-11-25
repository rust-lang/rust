// run-pass

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

fn main() {
    let mut t = (10, 10);

    let c = || {
        println!("{}", t.0);
    };

    // `c` only captures t.0, therefore mutating t.1 is allowed.
    let t1 = &mut t.1;

    c();
    *t1 = 20;
}
