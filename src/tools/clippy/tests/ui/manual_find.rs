#![allow(unused)]
#![warn(clippy::manual_find)]

fn vec_string(strings: Vec<String>) -> Option<String> {
    for s in strings {
        if s == String::new() {
            return Some(s);
        }
    }
    None
}

fn tuple(arr: Vec<(String, i32)>) -> Option<String> {
    for (s, _) in arr {
        if s == String::new() {
            return Some(s);
        }
    }
    None
}

fn main() {}
