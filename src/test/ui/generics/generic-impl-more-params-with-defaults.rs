use std::marker;

struct Heap;

struct Vec<T, A = Heap>(
    marker::PhantomData<(T,A)>);

impl<T, A> Vec<T, A> {
    fn new() -> Vec<T, A> {Vec(marker::PhantomData)}
}

fn main() {
    Vec::<isize, Heap, bool>::new();
    //~^ ERROR wrong number of type arguments
}
