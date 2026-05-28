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

#[rustc_dump_variances]
struct TestStruct<U,T:Setter<U>> { //~ ERROR [U: +, T: +]
    t: T, u: U
}

#[rustc_dump_variances]
enum TestEnum<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    Foo(T)
}

#[rustc_dump_variances]
struct TestContraStruct<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}

#[rustc_dump_variances]
struct TestBox<U,T:Getter<U>+Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}

pub fn main() { }
