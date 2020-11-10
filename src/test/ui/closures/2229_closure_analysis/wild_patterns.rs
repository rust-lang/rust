#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

// Test to ensure that we can handle cases where
// let statements create no bindings are intialized
// using a Place expression
//
// Note: Currently when feature `capture_disjoint_fields` is enabled
// we can't handle such cases. So the test so the test

struct Point {
    x: i32,
    y: i32,
}

fn wild_struct() {
    let p = Point { x: 10, y: 20 };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        // FIXME(arora-aman): Change `_x` to `_`
        let Point { x: _x, y: _ } = p;
        //~^ ERROR: Capturing p[(0, 0)] -> ImmBorrow
        //~| ERROR: Min Capture p[(0, 0)] -> ImmBorrow
    };

    c();
}

fn wild_tuple() {
    let t = (String::new(), 10);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        // FIXME(arora-aman): Change `_x` to `_`
        let (_x, _) = t;
        //~^ ERROR: Capturing t[(0, 0)] -> ByValue
        //~| ERROR: Min Capture t[(0, 0)] -> ByValue
    };

    c();
}

fn wild_arr() {
    let arr = [String::new(), String::new()];

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        // FIXME(arora-aman): Change `_x` to `_`
        let [_x, _] = arr;
        //~^ ERROR: Capturing arr[Index] -> ByValue
        //~| ERROR: Min Capture arr[] -> ByValue
    };

    c();
}

fn main() {}
