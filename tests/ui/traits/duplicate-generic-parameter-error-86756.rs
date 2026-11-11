// https://github.com/rust-lang/rust/issues/86756
//@ edition: 2015
trait Foo<T, T = T> {}
//~^ ERROR the name `T` is already used for a generic parameter in this item's generic parameters

fn eq<A, B>() {
    eq::<dyn, Foo>
    //~^ ERROR cannot find type `dyn` in this scope
    //~| ERROR missing generics for trait `Foo`
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}

fn main() {}
