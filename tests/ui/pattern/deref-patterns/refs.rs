//@ check-pass
#![feature(deref_patterns)]
#![expect(incomplete_features)]

fn foo(s: &String) -> i32 {
    match *s {
        "a" => 42,
        _ => -1,
    }
}

fn bar(s: Option<&&&&String>) -> i32 {
    match s {
        Some(&&&&"&&&&") => 1,
        _ => -1,
    }
}

fn main() {}
