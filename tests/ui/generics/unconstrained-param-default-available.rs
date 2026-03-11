//! Test that making use of parameter is suggested when a parameter with default type is available

struct S<U = i32> {
    _u: U,
}
impl<V> MyTrait for S {}
//~^ ERROR the type parameter `V` is not constrained by the impl trait, self type, or predicates

struct S2<T, U = i32> {
    _t: T,
    _u: U,
}
impl<T, V> MyTrait for S2<T> {}
//~^ ERROR the type parameter `V` is not constrained by the impl trait, self type, or predicates

trait MyTrait {}

fn main() {}
