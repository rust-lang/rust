//@ edition:2021
trait Trait {}

struct Foo1 {
    a: Trait,
    //~^ ERROR expected a type, found a trait
    b: u32,
}

struct Foo2 {
    a: i32,
    b: Trait,
    //~^ ERROR expected a type, found a trait
}

enum Enum1 {
    A(Trait),
    //~^ ERROR expected a type, found a trait
    B(u32),
}

enum Enum2 {
    A(u32),
    B(Trait),
    //~^ ERROR expected a type, found a trait
}

// Regression test for <https://github.com/rust-lang/rust/issues/138229>.
pub struct InWhereClause
where
    Trait:, {}
//~^ ERROR expected a type, found a trait

struct HasGenerics<T> {
    f: Trait,
    //~^ ERROR expected a type, found a trait
    t: T,
}


fn main() {}
