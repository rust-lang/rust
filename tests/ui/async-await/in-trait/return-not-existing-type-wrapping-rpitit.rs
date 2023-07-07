// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]

struct Wrapper<T>(T);

trait Foo {
    fn bar() -> Wrapper<Missing<impl Sized>>;
    //~^ ERROR: cannot find type `Missing` in this scope [E0412]
}

impl Foo for () {
    fn bar() -> Wrapper<i32> {
        Wrapper(0)
    }
}

fn main() {}
