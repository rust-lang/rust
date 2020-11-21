// Checks that we can build a clone shim for array with generic size.
// Regression test for issue #79269.
//
// build-pass
// compile-flags: -Zmir-opt-level=2 -Zvalidate-mir
#![feature(min_const_generics)]

#[derive(Clone)]
struct Array<T, const N: usize>([T; N]);

fn main() {
    let _ = Array([0u32, 1u32, 2u32]).clone();
}
