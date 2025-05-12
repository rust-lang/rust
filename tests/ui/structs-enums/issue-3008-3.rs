use std::marker;

enum E1 { V1(E2<E1>), }
enum E2<T> { V2(E2<E1>, marker::PhantomData<T>), }
//~^ ERROR recursive type `E2` has infinite size

impl E1 { fn foo(&self) {} }

fn main() {
}
