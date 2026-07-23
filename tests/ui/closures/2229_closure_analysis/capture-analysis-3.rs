//@ edition:2021

#![feature(rustc_attrs)]

#[derive(Debug)]
struct Child {
    c: String,
    d: String,
}

#[derive(Debug)]
struct Parent {
    b: Child,
}

fn main() {
    let mut a = Parent { b: Child {c: String::new(), d: String::new()} };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        let _x = a.b.c;
        //~^ NOTE: Capturing a[(0, 0),(0, 0)] -> ByValue
        //~| NOTE: a[(0, 0)] captured as ByValue here
        println!("{:?}", a.b);
        //~^ NOTE: Capturing a[(0, 0)] -> Immutable
        //~| NOTE: Min Capture a[(0, 0)] -> ByValue
        //~| NOTE: a[(0, 0)] used here
    };
}
