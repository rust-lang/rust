#![feature(auto_traits)]
#![feature(negative_impls)]

unsafe auto trait Trait {
    type Output; //~ ERROR E0380
}

fn call_method<T: Trait>(x: T) {}

fn main() {
    // ICE
    call_method(());
}
