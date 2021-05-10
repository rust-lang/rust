struct A<T> {
//~^ ERROR recursive type `A` has infinite size
    x: T,
    y: B<T>,
}

struct B<T> {
//~^ ERROR recursive type `B` has infinite size
    z: A<T>
}

struct C<T> {
//~^ ERROR recursive type `C` has infinite size
    x: T,
    y: Option<Option<D<T>>>,
}

struct D<T> {
//~^ ERROR recursive type `D` has infinite size
    z: Option<Option<C<T>>>,
}

fn main() {}
