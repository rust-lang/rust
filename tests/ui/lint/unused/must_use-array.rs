#![deny(unused_must_use)]

#[must_use]
#[derive(Clone, Copy)]
struct S;

struct A;

#[must_use]
trait T {}

impl T for A {}

fn empty() -> [S; 0] {
    []
}

fn singleton() -> [S; 1] {
    [S]
}

fn many() -> [S; 4] {
    [S, S, S, S]
}

fn array_of_impl_trait() -> [impl T; 2] {
    [A, A]
}

fn impl_array() -> [(u8, Box<dyn T>); 2] {
    [(0, Box::new(A)), (0, Box::new(A))]
}

fn array_of_arrays_of_arrays() -> [[[S; 1]; 2]; 1] {
    [[[S], [S]]]
}

fn usize_max() -> [S; usize::MAX] {
    [S; usize::MAX]
}

fn main() {
    empty(); // ok
    singleton(); //~ ERROR unused array of `S` that must be used
    many(); //~ ERROR unused array of `S` that must be used
    ([S], 0, ()); //~ ERROR unused array of `S` in tuple element 0 that must be used
    array_of_impl_trait(); //~ ERROR unused array of implementers of `T` that must be used
    impl_array();
    //~^ ERROR unused array of boxed `T` trait objects in tuple element 1 that must be used
    array_of_arrays_of_arrays();
    //~^ ERROR unused array of arrays of arrays of `S` that must be used
    usize_max();
    //~^ ERROR unused array of `S` that must be used
}
