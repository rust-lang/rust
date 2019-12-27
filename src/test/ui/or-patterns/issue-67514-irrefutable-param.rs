// Check that we don't ICE for irrefutable or-patterns in function parameters

// check-pass

#![feature(or_patterns)]

fn foo((Some(_) | None): Option<u32>) {}

fn main() {
    foo(None);
}
