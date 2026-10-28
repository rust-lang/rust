//@ revisions: edition2018 edition2021
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021

#![feature(ergonomic_clones, rustc_attrs, stmt_expr_attributes)]
#![expect(incomplete_features, internal_features)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

#[derive(Clone)]
struct Bar(String);

impl UseCloned for Bar {}

#[derive(Clone)]
struct Baz(Foo);

impl UseCloned for Baz {}

fn main() {
    let foo = Foo;

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = foo.use;
        //~^ NOTE Capturing foo[] -> ByUse
        //~| NOTE Min Capture foo[] -> ByUse
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = foo.use;
        //~^ NOTE Capturing foo[] -> ByUse
        let _f = foo;
        //~^ NOTE Capturing foo[] -> ByValue
        //~| NOTE Min Capture foo[] -> ByValue
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = foo.use;
        //~^ NOTE Capturing foo[] -> ByUse
        let _f = &foo;
        //~^ NOTE Capturing foo[] -> Immutable
        //~| NOTE Min Capture foo[] -> Immutable
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = (&foo).use;
        //~^ NOTE Capturing foo[] -> Immutable
        //~| NOTE Min Capture foo[] -> Immutable
    };

    let tup = (foo, false);

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        //[edition2018]~| NOTE tup[] used here
        //[edition2018]~| NOTE Min Capture tup[] -> Immutable
        let _f = tup.0.use;
        //~^ NOTE Capturing tup[(0, 0)] -> ByUse
        //[edition2018]~| NOTE tup[] captured as Immutable here
        //[edition2021]~| NOTE Min Capture tup[(0, 0)] -> ByUse
        let _b = tup.1.use;
        //~^ NOTE Capturing tup[(1, 0)] -> ByUse
        //[edition2021]~| NOTE Min Capture tup[(1, 0)] -> ByUse
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = tup.0.use;
        //~^ NOTE Capturing tup[(0, 0)] -> ByUse
        let _t = tup;
        //~^ NOTE Capturing tup[] -> ByValue
        //~| NOTE Min Capture tup[] -> ByValue
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = tup.0.use;
        //~^ NOTE Capturing tup[(0, 0)] -> ByUse
        let _t = &tup;
        //~^ NOTE Capturing tup[] -> Immutable
        //~| NOTE Min Capture tup[] -> Immutable
    };

    let tup2 = (foo, String::new());

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        //[edition2018]~| NOTE tup2[] used here
        //[edition2018]~| NOTE Min Capture tup2[] -> ByValue
        let _f = tup2.0.use;
        //~^ NOTE Capturing tup2[(0, 0)] -> ByUse
        //[edition2021]~| NOTE Min Capture tup2[(0, 0)] -> ByUse
        let _s = tup2.1;
        //~^ NOTE Capturing tup2[(1, 0)] -> ByValue
        //[edition2018]~| NOTE tup2[] captured as ByValue here
        //[edition2021]~| NOTE Min Capture tup2[(1, 0)] -> ByValue
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = tup2.0.use;
        //~^ NOTE Capturing tup2[(0, 0)] -> ByUse
        let _t = tup2;
        //~^ NOTE Capturing tup2[] -> ByValue
        //~| NOTE Min Capture tup2[] -> ByValue
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = tup2.0.use;
        //~^ NOTE Capturing tup2[(0, 0)] -> ByUse
        let _t = &tup2;
        //~^ NOTE Capturing tup2[] -> Immutable
        //~| NOTE Min Capture tup2[] -> Immutable
    };

    let bar = Bar(String::new());

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _b = bar.use;
        //~^ NOTE Capturing bar[] -> ByUse
        //~| NOTE Min Capture bar[] -> ByUse
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _s = &bar.0;
        //~^ NOTE Capturing bar[(0, 0)] -> Immutable
        //~| NOTE bar[] captured as Immutable here
        let _b = bar.use;
        //~^ NOTE Capturing bar[] -> ByUse
        //~| NOTE bar[] used here
        //~| NOTE Min Capture bar[] -> Immutable
    };

    let baz = Baz(Foo);

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = baz.0.use;
        //~^ NOTE Capturing baz[(0, 0)] -> ByUse
        //~| NOTE baz[] captured as Immutable here
        let _b = baz.use;
        //~^ NOTE Capturing baz[] -> ByUse
        //~| NOTE baz[] used here
        //~| NOTE Min Capture baz[] -> Immutable
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = baz.use.0.use;
        //~^ NOTE Capturing baz[] -> ByUse
        //~| NOTE Min Capture baz[] -> ByUse
        let _b = baz.use;
        //~^ NOTE Capturing baz[] -> ByUse
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = baz.0;
        //~^ NOTE Capturing baz[(0, 0)] -> ByValue
        //~| NOTE baz[] captured as ByValue here
        let _b = baz.use;
        //~^ NOTE Capturing baz[] -> ByUse
        //~| NOTE baz[] used here
        //~| NOTE Min Capture baz[] -> ByValue
    };

    let _ = #[rustc_capture_analysis]
    || {
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        let _f = baz.0.use;
        //~^ NOTE Capturing baz[(0, 0)] -> ByUse
        let _b = baz;
        //~^ NOTE Capturing baz[] -> ByValue
        //~| NOTE Min Capture baz[] -> ByValue
    };
}
