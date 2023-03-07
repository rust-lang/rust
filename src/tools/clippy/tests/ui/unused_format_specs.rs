// run-rustfix

#![warn(clippy::unused_format_specs)]
#![allow(unused)]

fn main() {
    let f = 1.0f64;
    println!("{:.}", 1.0);
    println!("{f:.} {f:.?}");

    println!("{:.}", 1);
}

fn should_not_lint() {
    let f = 1.0f64;
    println!("{:.1}", 1.0);
    println!("{f:.w$} {f:.*?}", 3, w = 2);
}
