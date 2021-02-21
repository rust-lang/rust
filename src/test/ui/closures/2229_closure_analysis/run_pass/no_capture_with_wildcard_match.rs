#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

fn main() {
    let foo = [1, 2, 3];
    let c = #[rustc_capture_analysis] || {
        //~^ ERROR: attributes on expressions are experimental
        //~| ERROR: First Pass analysis includes:
        match foo { _ => () };
    };
}
