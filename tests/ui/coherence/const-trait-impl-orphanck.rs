//! Regression test for #149703.
#![feature(const_trait_impl)]
trait Z {
    type Assoc;
}
struct A;
impl<T: const FnOnce()> Z for T {
    type Assoc = ();
}
impl<T> From<<A as Z>::Assoc> for T {}
//~^ ERROR type parameter `T` must be used as an argument to some local type

fn main() {}
