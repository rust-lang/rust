#![allow(dead_code)]
#![feature(rustc_attrs)]

// Check that bounds on type parameters (other than `Self`) do not
// influence variance.

trait Getter<T> {
    fn get(&self) -> T;
}

trait Setter<T> {
    fn get(&self, _: T);
}

#[rustc_variance]
struct TestStruct<U,T:Setter<U>> { //~ ERROR [U: +, T: +]
    t: T, u: U
}

#[rustc_variance]
enum TestEnum<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    Foo(T)
}

#[rustc_variance]
struct TestContraStruct<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}

#[rustc_variance]
struct TestBox<U,T:Getter<U>+Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}

pub fn main() { }
