#![feature(existential_type)]

fn main() {}

existential type WrongGeneric<T: 'static>: 'static;
//~^ ERROR: at least one trait must be specified

fn wrong_generic<U: 'static, V: 'static>(_: U, v: V) -> WrongGeneric<U> {
//~^ ERROR type parameter `V` is part of concrete type but not used in parameter list
    v
}
