//@ run-pass

// Test that vec is now covariant in its argument type.

#![allow(dead_code)]

fn foo<'a,'b>(v1: Vec<&'a i32>, v2: Vec<&'b i32>) -> i32 {
    bar(v1, v2).cloned().unwrap_or(0) // only type checks if we can intersect 'a and 'b
}

fn bar<'c>(v1: Vec<&'c i32>, v2: Vec<&'c i32>) -> Option<&'c i32> {
    v1.get(0).cloned().or_else(|| v2.get(0).cloned())
}

fn main() {
    let x = 22;
    let y = 44;
    assert_eq!(foo(vec![&x], vec![&y]), 22);
    assert_eq!(foo(vec![&y], vec![&x]), 44);
}
