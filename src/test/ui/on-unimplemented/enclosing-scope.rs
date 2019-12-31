// Test scope annotations from `enclosing_scope` parameter

#![feature(rustc_attrs)]

#[rustc_on_unimplemented(enclosing_scope="in this scope")]
trait Trait{}

struct Foo;

fn f<T: Trait>(x: T) {}

fn main() {
    let x = || {
        f(Foo{}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
        let y = || {
            f(Foo{}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
        };
    };

    {
        {
            f(Foo{}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
        }
    }

    f(Foo{}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
}
