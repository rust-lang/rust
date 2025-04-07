//@check-pass

#![allow(clippy::redundant_pattern_matching)]

struct S<'a> {
    s: &'a str,
}

fn foo() -> Option<S<'static>> {
    if let Some(_) = Some(0) {
        Some(S { s: "xyz" })
    } else {
        None
    }
}

fn main() {}
