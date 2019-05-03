#![feature(existential_type)]

fn main() {}

existential type Underconstrained<T: std::fmt::Debug>: 'static;
//~^ ERROR `U` doesn't implement `std::fmt::Debug`
//~^^ ERROR: at least one trait must be specified

// not a defining use, because it doesn't define *all* possible generics
fn underconstrained<U>(_: U) -> Underconstrained<U> {
    5u32
}

existential type Underconstrained2<T: std::fmt::Debug>: 'static;
//~^ ERROR `V` doesn't implement `std::fmt::Debug`
//~^^ ERROR: at least one trait must be specified

// not a defining use, because it doesn't define *all* possible generics
fn underconstrained2<U, V>(_: U, _: V) -> Underconstrained2<V> {
    5u32
}
