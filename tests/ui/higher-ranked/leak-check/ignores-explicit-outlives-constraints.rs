// This checks whether the leak check during coherence considers explicit outlives
// bounds. If it did, this would compile. Right now, we intentionally error here.
// See #163267.

trait MayRequireStatic {}
// `for<'a> &'a i32: MayRequireStatic` does not hold
impl<'a: 'static> MayRequireStatic for &'a i32 {}
impl<'a> MayRequireStatic for &'a u32 {}

trait Trait {}

impl Trait for i32 {}
impl<T> Trait for T where for<'a> &'a T: MayRequireStatic {}
//~^ ERROR conflicting implementations of trait `Trait` for type `i32`

fn is_trait<T: Trait>() {}

fn main() {
    is_trait::<i32>();
    is_trait::<u32>();
}
