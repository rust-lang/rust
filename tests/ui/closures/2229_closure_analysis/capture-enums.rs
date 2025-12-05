//@ edition:2021

#![feature(rustc_attrs)]

enum Info {
    Point(i32, i32, String),
    Meta(String, Vec<(i32, i32)>)
}

fn multi_variant_enum() {
    let point = Info::Point(10, -10, "1".into());

    let vec = Vec::new();
    let meta = Info::Meta("meta".into(), vec);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        if let Info::Point(_, _, str) = point {
            //~^ NOTE: Capturing point[] -> Immutable
            //~| NOTE: Capturing point[(2, 0)] -> ByValue
            //~| NOTE: Min Capture point[] -> ByValue
            println!("{}", str);
        }

        if let Info::Meta(_, v) = meta {
            //~^ NOTE: Capturing meta[] -> Immutable
            //~| NOTE: Capturing meta[(1, 1)] -> ByValue
            //~| NOTE: Min Capture meta[] -> ByValue
            println!("{:?}", v);
        }
    };

    c();
}

enum SingleVariant {
    Point(i32, i32, String),
}

fn single_variant_enum() {
    let point = SingleVariant::Point(10, -10, "1".into());

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        let SingleVariant::Point(_, _, str) = point;
        //~^ NOTE: Capturing point[(2, 0)] -> ByValue
        //~| NOTE: Min Capture point[(2, 0)] -> ByValue
        println!("{}", str);
    };

    c();
}

fn main() {}
