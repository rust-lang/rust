//@ known-bug: #149703
#![feature(const_trait_impl)]
trait Z {
    type Assoc;
}
struct A;
impl<T: const FnOnce()> Z for T {
    type Assoc = ();
}
impl<T> From<<A as Z>::Assoc> for T {}

fn main() {}
