// Regression test for issue 67856

#![feature(unboxed_closures)]
#![feature(type_alias_impl_trait)]
#![feature(fn_traits)]

trait MyTrait {}
impl MyTrait for () {}

impl<F> FnOnce<()> for &F {
    //~^ ERROR type parameter `F` must be used
    type Output = impl MyTrait;
    extern "rust-call" fn call_once(self, _: ()) -> Self::Output {}
}
fn main() {}
