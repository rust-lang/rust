//! Regression test for https://github.com/rust-lang/rust/issues/37164.

//@ check-pass

use std::collections::HashMap;

fn check(_: &str) -> bool {
    false
}

fn chain<'a>(some_key: &'a str) -> HashMap<&'a str, Vec<usize>> {
    let map = HashMap::new();
    let _ = map.get(&some_key);

    let key_ref = map.keys().next().unwrap();
    if check(key_ref) {}

    map
}

fn main() {
    drop(chain("key"));
}
