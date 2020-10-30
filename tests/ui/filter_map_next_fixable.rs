// run-rustfix

#![warn(clippy::all, clippy::pedantic)]

fn main() {
    let a = ["1", "lol", "3", "NaN", "5"];

    let element: Option<i32> = a.iter().filter_map(|s| s.parse().ok()).next();
    assert_eq!(element, Some(1));
}
