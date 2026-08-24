// Test that an or-pattern works with a wild pattern. This tests two things:
//
//  1) The Wild pattern should cause the pattern to always succeed.
//  2) or-patterns should work with simplifyable patterns.

//@ run-pass

// `Some(0 | _)` is deliberate: this tests the runtime behavior of exactly that mix.
#![allow(unreachable_patterns)]

pub fn test(x: Option<usize>) -> bool {
    match x {
        Some(0 | _) => true,
        _ => false,
    }
}

fn main() {
    assert!(test(Some(42)));
    assert!(!test(None));
}
