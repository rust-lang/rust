#![feature(optin_builtin_traits)]

unsafe auto trait Trait {
//~^ ERROR E0380
    fn method(&self) {
        println!("Hello");
    }
}

fn call_method<T: Trait>(x: T) {
    x.method();
}

fn main() {
    // ICE
    call_method(());
}
