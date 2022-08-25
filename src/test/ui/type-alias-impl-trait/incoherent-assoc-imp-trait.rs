// Regression test for issue 67856

#![feature(unboxed_closures)]
#![feature(type_alias_impl_trait)]
#![feature(fn_traits)]

trait MyTrait {}
impl MyTrait for () {}

impl<F> FnOnce<()> for &F {
    //~^ ERROR type parameter `F` as argument to a fundamental type
    // | must be used as the type parameter for some local type
    // | (e.g., `MyStruct<F>`)
    type Output = impl MyTrait;
    extern "rust-call" fn call_once(self, _: ()) -> Self::Output {}
}
fn main() {}
