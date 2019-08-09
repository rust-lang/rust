//~ ERROR

#![feature(optin_builtin_traits)]

unsafe auto trait Trait {
//~^ ERROR E0380
    type Output;
}

fn call_method<T: Trait>(x: T) {}

fn main() {
    // ICE
    call_method(());
}
