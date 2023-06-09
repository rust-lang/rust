// check-pass

#![deny(unused_lifetimes)]
trait Trait2 {
    type As;
}

// we should not warn about an unused lifetime about code generated from this proc macro here
#[derive(Clone)]
struct ShimMethod4<T: Trait2 + 'static>(pub &'static dyn for<'s> Fn(&'s mut T::As));

pub fn main() {}
