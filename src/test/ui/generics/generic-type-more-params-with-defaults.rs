use std::marker;

struct Heap;

struct Vec<T, A = Heap>(
    marker::PhantomData<(T,A)>);

fn main() {
    let _: Vec<isize, Heap, bool>;
    //~^ ERROR wrong number of type arguments: expected at most 2, found 3 [E0107]
}
