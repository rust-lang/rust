// Regression test for https://github.com/rust-lang/rust/issues/152990.
//@ compile-flags: -Znext-solver
#![feature(checked_type_aliases)]

struct Inv;
enum Foo<F: Fn()> {
    Bar,
    Var(F),
}
type Supertype = Foo<dyn Fn()>;
//~^ ERROR the size for values of type `(dyn Fn() + 'static)` cannot be known

fn main() {
    let x: Inv = todo!();
    match x {
        Foo::Var(Supertype::Bar) => {}
        //~^ ERROR mismatched types
        //~| ERROR the size for values of type `(dyn Fn() + 'static)` cannot be known
        //~| ERROR expected `Fn()` signature, found `Foo<dyn Fn()>`
        _ => {}
    }
}
