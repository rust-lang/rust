//check-pass
#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

fn main() {
    let mut x = 1;
    let c = || {
        drop(&mut x);
        match x { _ => () }
    };
}
