// Test scope annotations from `parent_label` parameter

#![feature(rustc_attrs)]

#[rustc_on_unimplemented(parent_label = "in this scope")]
trait Trait {}

struct Foo;

fn f<T: Trait>(x: T) {}

fn main() {
    let x = || {
        f(Foo {}); //~ ERROR trait `Trait` is not implemented for `Foo`
        let y = || {
            f(Foo {}); //~ ERROR trait `Trait` is not implemented for `Foo`
        };
    };

    {
        {
            f(Foo {}); //~ ERROR trait `Trait` is not implemented for `Foo`
        }
    }

    f(Foo {}); //~ ERROR trait `Trait` is not implemented for `Foo`
}
