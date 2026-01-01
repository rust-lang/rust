//@ run-pass
#![feature(try_as_dyn)]

use std::fmt::Debug;

// Look ma, no `T: Debug`
fn debug_format_with_try_as_dyn<T: 'static>(t: &T) -> String {
    match std::any::try_as_dyn::<_, dyn Debug>(t) {
        Some(d) => format!("{d:?}"),
        None => "default".to_string()
    }
}

// Test that downcasting to a dyn trait works as expected
fn main() {
    #[allow(dead_code)]
    #[derive(Debug)]
    struct A {
        index: usize
    }
    let a = A { index: 42 };
    let result = debug_format_with_try_as_dyn(&a);
    assert_eq!("A { index: 42 }", result);

    struct B {}
    let b = B {};
    let result = debug_format_with_try_as_dyn(&b);
    assert_eq!("default", result);
}
