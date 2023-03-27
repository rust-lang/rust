// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
//~^ WARN the feature `return_position_impl_trait_in_trait` is incomplete

trait MyTrait {
    fn foo(&self) -> impl Sized;
    fn bar(&self) -> impl Sized;
}

impl MyTrait for i32 {
    //~^ ERROR not all trait items implemented, missing: `foo`
    fn bar(&self) -> impl Sized {
        self.foo()
    }
}

fn main() {}
