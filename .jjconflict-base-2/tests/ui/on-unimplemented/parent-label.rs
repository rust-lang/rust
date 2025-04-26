// Test scope annotations from `parent_label`, including when it is in a later attribute.

#![feature(rustc_attrs)]

#[rustc_on_unimplemented(label = "unsatisfied trait bound")]
#[rustc_on_unimplemented(parent_label = "in this scope")]
#[rustc_on_unimplemented(parent_label = "ignored parent label")]
//~^ WARN `parent_label` is ignored due to previous definition of `parent_label`
trait Trait {}

struct Foo;

fn f<T: Trait>(x: T) {}

fn main() {
    let x = || {
        f(Foo {}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
        let y = || {
            f(Foo {}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
        };
    };

    {
        {
            f(Foo {}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
        }
    }

    f(Foo {}); //~ ERROR the trait bound `Foo: Trait` is not satisfied
}
