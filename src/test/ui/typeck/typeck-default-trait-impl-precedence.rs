// Test that declaring that `&T` is `Defaulted` if `T:Signed` implies
// that other `&T` is NOT `Defaulted` if `T:Signed` does not hold. In
// other words, the auto impl only applies if there are no existing
// impls whose types unify.

#![feature(optin_builtin_traits)]

auto trait Defaulted { }
impl<'a,T:Signed> Defaulted for &'a T { }
impl<'a,T:Signed> Defaulted for &'a mut T { }
fn is_defaulted<T:Defaulted>() { }

trait Signed { }
impl Signed for i32 { }

fn main() {
    is_defaulted::<&'static i32>();
    is_defaulted::<&'static u32>();
    //~^ ERROR `u32: Signed` is not satisfied
}
