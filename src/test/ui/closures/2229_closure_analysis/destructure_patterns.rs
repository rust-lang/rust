#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

// Test to ensure Index projections are handled properly during capture analysis
// The array should be moved in entirety, even though only some elements are used.
fn arrays() {
    let arr: [String; 5] = [format!("A"), format!("B"), format!("C"), format!("D"), format!("E")];

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        let [a, b, .., e] = arr;
        //~^ ERROR: Capturing arr[Index] -> ByValue
        //~| ERROR: Min Capture arr[] -> ByValue
        assert_eq!(a, "A");
        assert_eq!(b, "B");
        assert_eq!(e, "E");
    };

    c();
}

struct Point {
    x: i32,
    y: i32,
    id: String,
}

fn structs() {
    let mut p = Point { x: 10, y: 10, id: String::new() };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        let Point { x: ref mut x, y: _, id: moved_id } = p;
        //~^ ERROR: Capturing p[(0, 0)] -> MutBorrow
        //~| ERROR: Capturing p[(2, 0)] -> ByValue
        //~| ERROR: Min Capture p[(0, 0)] -> MutBorrow
        //~| ERROR: Min Capture p[(2, 0)] -> ByValue

        println!("{}, {}", x, moved_id);
    };
    c();
}

fn tuples() {
    let mut t = (10, String::new(), (String::new(), 42));

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        let (ref mut x, ref ref_str, (moved_s, _)) = t;
        //~^ ERROR: Capturing t[(0, 0)] -> MutBorrow
        //~| ERROR: Capturing t[(1, 0)] -> ImmBorrow
        //~| ERROR: Capturing t[(2, 0),(0, 0)] -> ByValue
        //~| ERROR: Min Capture t[(0, 0)] -> MutBorrow
        //~| ERROR: Min Capture t[(1, 0)] -> ImmBorrow
        //~| ERROR: Min Capture t[(2, 0),(0, 0)] -> ByValue

        println!("{}, {} {}", x, ref_str, moved_s);
    };
    c();
}

fn main() {}
