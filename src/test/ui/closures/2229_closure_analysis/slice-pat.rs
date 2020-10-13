#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

// Test to ensure Index projections are handled properly during capture analysis
//
// The array should be moved in entirety, even though only some elements are used.

fn main() {
    let arr : [String; 5] = [
        format!("A"),
        format!("B"),
        format!("C"),
        format!("D"),
        format!("E")
    ];

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
        || {
            let [a, b, .., e] = arr;
            assert_eq!(a, "A");
            assert_eq!(b, "B");
            assert_eq!(e, "E");
        };

    c();
}
