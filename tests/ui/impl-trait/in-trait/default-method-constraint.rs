// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

// This didn't work in the previous default RPITIT method hack attempt

#![feature(return_position_impl_trait_in_trait)]
//~^ WARN the feature `return_position_impl_trait_in_trait` is incomplete

trait Foo {
    fn bar(x: bool) -> impl Sized {
        if x {
            let _: u32 = Self::bar(!x);
        }
        Default::default()
    }
}

fn main() {}
