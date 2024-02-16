//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Check that inherent associated types are dispatched on the concrete Self type.

struct Select<T>(T);

impl Select<u8> {
    type Projection = ();
}

impl Select<String> {
    type Projection = bool;
}

struct Choose<T>(T);
struct NonCopy;

impl<T: Copy> Choose<T> {
    type Result = Vec<T>;
}

impl Choose<NonCopy> {
    type Result = ();
}

fn main() {
    let _: Select<String>::Projection = false;
    let _: Select<u8>::Projection = ();

    let _: Choose<NonCopy>::Result = ();
    let _: Choose<&str>::Result = vec!["…"]; // regression test for issue #108957
}

// Test if we use the correct `ParamEnv` when proving obligations.

pub fn parameterized<T: Copy>(x: T) {
    let _: Choose<T>::Result = vec![x];
}
