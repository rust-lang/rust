//check-pass
#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

fn main() {
    let mut x = 0;
    let c = || {
        &mut x; // mutable borrow of `x`
        match x { _ => () } // fake read of `x`
    };
}
