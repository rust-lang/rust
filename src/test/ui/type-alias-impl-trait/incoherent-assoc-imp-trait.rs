// Regression test for issue 67856

#![feature(unboxed_closures)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![feature(fn_traits)]

trait MyTrait {}
impl MyTrait for () {}

impl<F> FnOnce<()> for &F {
    //~^ ERROR conflicting implementations
    //~| ERROR type parameter `F` must be used
    type Output = impl MyTrait;
    extern "rust-call" fn call_once(self, _: ()) -> Self::Output {}
}
fn main() {}
