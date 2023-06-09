struct A<T> {
//~^ ERROR recursive type `A` has infinite size
    x: T,
    y: A<A<T>>,
}

struct B {
    z: A<usize>
}

fn main() {}
