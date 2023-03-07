use std::fmt::Debug;

// check-pass

fn bar() -> impl Debug {}

fn baz(b: bool) -> Option<impl Debug> {
    match b {
        true => baz(false),
        false => Some(bar()),
    }
}

fn main() {}
