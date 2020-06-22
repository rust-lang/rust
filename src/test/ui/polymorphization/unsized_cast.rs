// build-fail
// compile-flags: -Zpolymorphize-errors
#![feature(fn_traits, unboxed_closures)]

// This test checks that the polymorphization analysis considers a closure
// as using all generic parameters if it does an unsizing cast.

fn foo<T: Default>() {
    let _: T = Default::default();
    (|| Box::new(|| {}) as Box<dyn Fn()>)();
    //~^ ERROR item has unused generic parameters
    //~^^ ERROR item has unused generic parameters
}

fn foo2<T: Default>() {
    let _: T = Default::default();
    (|| {
        let call: extern "rust-call" fn(_, _) = Fn::call;
        call(&|| {}, ());
        //~^ ERROR item has unused generic parameters
    })();
}

fn main() {
    foo::<u32>();
    foo2::<u32>();
}
