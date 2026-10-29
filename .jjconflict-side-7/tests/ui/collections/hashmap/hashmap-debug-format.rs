//! Regression test for <https://github.com/rust-lang/rust/issues/3559>.
//! Test hashmap debug format doesn't ICE.
//@ run-pass

use std::collections::HashMap;

fn check_strs(actual: &str, expected: &str) -> bool {
    if actual != expected {
        println!("Found {}, but expected {}", actual, expected);
        return false;
    }
    return true;
}

pub fn main() {
    let mut table = HashMap::new();
    table.insert("one".to_string(), 1);
    table.insert("two".to_string(), 2);
    assert!(check_strs(&format!("{:?}", table), "{\"one\": 1, \"two\": 2}") ||
            check_strs(&format!("{:?}", table), "{\"two\": 2, \"one\": 1}"));
}
