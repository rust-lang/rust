//@ run-pass
#![feature(downcast_trait)]

use std::fmt::Debug;

// Look ma, no `T: Debug`
fn downcast_debug_format<T: 'static>(t: &T) -> String {
    match std::any::downcast_trait::<_, dyn Debug>(t) {
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
    let result = downcast_debug_format(&a);
    assert_eq!("A { index: 42 }", result);

    struct B {}
    let b = B {};
    let result = downcast_debug_format(&b);
    assert_eq!("default", result);
}
