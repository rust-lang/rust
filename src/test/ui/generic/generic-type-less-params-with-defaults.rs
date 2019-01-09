use std::marker;

struct Heap;

struct Vec<T, A = Heap>(
    marker::PhantomData<(T,A)>);

fn main() {
    let _: Vec;
    //~^ ERROR wrong number of type arguments: expected at least 1, found 0 [E0107]
}
