#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

// Ensure that capture analysis results in arrays being completely captured.
fn main() {
    let mut m = [1, 2, 3, 4, 5];

    let mut c = #[rustc_capture_analysis]
    || {
        //~^ ERROR: attributes on expressions are experimental
        m[0] += 10;
        m[1] += 40;
    };

    c();
}
