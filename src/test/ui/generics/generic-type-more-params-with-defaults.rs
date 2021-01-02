use std::marker;

struct Heap;

struct Vec<T, A = Heap>(
    marker::PhantomData<(T,A)>);

fn main() {
    let _: Vec<isize, Heap, bool>;
    //~^ ERROR this struct takes at most 2 type arguments but 3 type arguments were supplied
}
