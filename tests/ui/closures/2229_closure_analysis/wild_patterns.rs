//@ edition:2021

#![feature(rustc_attrs)]

// Test to ensure that we can handle cases where
// let statements create no bindings are initialized
// using a Place expression
//
// Note: Currently when feature `capture_disjoint_fields` is enabled
// we can't handle such cases. So the test current use `_x` instead of
// `_` until the issue is resolved.
// Check rust-lang/project-rfc-2229#24 for status.

struct Point {
    x: i32,
    y: i32,
}

fn wild_struct() {
    let p = Point { x: 10, y: 20 };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        // FIXME(arora-aman): Change `_x` to `_`
        let Point { x: _x, y: _ } = p;
        //~^ NOTE: Capturing p[(0, 0)] -> Immutable
        //~| NOTE: Min Capture p[(0, 0)] -> Immutable
    };

    c();
}

fn wild_tuple() {
    let t = (String::new(), 10);

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        // FIXME(arora-aman): Change `_x` to `_`
        let (_x, _) = t;
        //~^ NOTE: Capturing t[(0, 0)] -> ByValue
        //~| NOTE: Min Capture t[(0, 0)] -> ByValue
    };

    c();
}

fn wild_arr() {
    let arr = [String::new(), String::new()];

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        // FIXME(arora-aman): Change `_x` to `_`
        let [_x, _] = arr;
        //~^ NOTE: Capturing arr[Index] -> ByValue
        //~| NOTE: Min Capture arr[] -> ByValue
    };

    c();
}

fn main() {}
