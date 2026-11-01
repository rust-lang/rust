use std::marker;

struct Heap;

struct Vec<T, A = Heap>(
    marker::PhantomData<(T,A)>);

fn main() {
    let _: Vec<isize, Heap, bool>;
    //~^ ERROR struct takes at most 2 generic arguments but 3 generic arguments
}
