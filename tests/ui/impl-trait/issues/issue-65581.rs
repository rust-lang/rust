// check-pass
// ignore-compare-mode-chalk

#![allow(dead_code)]

trait Trait1<T, U> {
    fn f1(self) -> U;
}

trait Trait2 {
    type T;
    type U: Trait2<T = Self::T>;
    fn f2(f: impl FnOnce(&Self::U));
}

fn f3<T: Trait2>() -> impl Trait1<T, T::T> {
    Struct1
}

struct Struct1;

impl<T: Trait2> Trait1<T, T::T> for Struct1 {
    fn f1(self) -> T::T {
        unimplemented!()
    }
}

fn f4<T: Trait2>() {
    T::f2(|_| {
        f3::<T::U>().f1();
    });
}

fn main() {}
