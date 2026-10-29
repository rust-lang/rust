#![feature(coverage_attribute)]
//@ edition: 2024
//@ revisions: zero one many
//@[one] ignore-coverage-map
//@[many] ignore-coverage-map

// Basic test of `for` loops.

fn for_loop(items: &[&str]) {
    say("hello");

    for item in items {
        say(item);
    }

    for item in items {
        say(item)
    }

    say("goodbye");
}

#[coverage(off)]
fn main() {
    let items = cfg_select!(
        zero => &[],
        one => &["one"],
        many => &["one", "two", "three"],
    );
    for_loop(items);
}

#[coverage(off)]
fn say(msg: &str) {
    println!("{msg}");
}
