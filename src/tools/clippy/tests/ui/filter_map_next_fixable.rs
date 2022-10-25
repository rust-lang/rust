// run-rustfix

#![feature(custom_inner_attributes)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(unused)]

fn main() {
    let a = ["1", "lol", "3", "NaN", "5"];

    let element: Option<i32> = a.iter().filter_map(|s| s.parse().ok()).next();
    assert_eq!(element, Some(1));
}

fn msrv_1_29() {
    #![clippy::msrv = "1.29"]

    let a = ["1", "lol", "3", "NaN", "5"];
    let _: Option<i32> = a.iter().filter_map(|s| s.parse().ok()).next();
}

fn msrv_1_30() {
    #![clippy::msrv = "1.30"]

    let a = ["1", "lol", "3", "NaN", "5"];
    let _: Option<i32> = a.iter().filter_map(|s| s.parse().ok()).next();
}
