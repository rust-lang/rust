//@ run-pass
//! Test using `#[splat]` on tuples with complex generic types inside the splatted tuple.
//! This fills the FIXME in splat-generics-everywhere.rs:
//! "also add generics inside the splatted tuple"

#![allow(incomplete_features)]
#![feature(splat)]

// Vec<T> and Option<U> inside splatted tuple
fn nested_generic<T, U>(#[splat] _: (Vec<T>, Option<U>)) {}

// Box<T> inside splatted tuple
fn box_generic<T>(#[splat] _: (Box<T>, u32)) {}

// Multiple complex generics
fn multi_generic<T, U, V>(#[splat] _: (Vec<T>, Option<U>, Box<V>)) {}

fn main() {
    nested_generic(vec![1u32, 2u32], Some(2i8));
    nested_generic(vec![1, 2, 3], None::<i8>);
    nested_generic::<u32, &str>(vec![], Some("hello"));

    box_generic(Box::new(1u32), 42u32);
    box_generic(Box::new("hello"), 1u32);

    multi_generic(vec![1u32], Some(2i8), Box::new(3.0f64));
    multi_generic::<u32, i8, &str>(vec![], None::<i8>, Box::new("hello"));
}
