fn testing_mytrait<T>(_: T) where T: MyTrait<u8> {}

trait MyTrait<T> {}

impl<T, U> MyTrait<U> for Vec<T> where T: MyTrait<U> {}
impl<T, U> MyTrait<U> for &T where T: MyTrait<U> {}
impl<T, U> MyTrait<U> for [T] where T: MyTrait<U> {}
impl<T, V> MyTrait<V> for [T;2] where T: MyTrait<V> {}

impl<T1, U> MyTrait<U> for (T1,) where T1: MyTrait<U> {}
impl<T1, T2, U> MyTrait<U> for (T1, T2) where T1: MyTrait<U> {}
impl<T1, T2, T3, U> MyTrait<U> for (T1, T2, T3) where T1: MyTrait<U> {}
impl<T1, T2, T3, T4, U> MyTrait<U> for (T1, T2, T3, T4) where T1: MyTrait<U> {}
impl<T1, T2, T3, T4, T5, U> MyTrait<U> for (T1, T2, T3, T4, T5) where T1: MyTrait<U> {}

struct Foo;

fn main() {
    testing_mytrait(Foo);
    //~^ ERROR the trait bound `Foo: MyTrait<u8>` is not satisfied [E0277]
}
