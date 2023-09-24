#![feature(rustc_attrs)]
#![feature(negative_impls)]

#[rustc_auto_trait]
unsafe trait Trait {
    fn method(&self) { //~ ERROR E0380
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
