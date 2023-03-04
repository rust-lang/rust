// build-fail
// compile-flags:-Zpolymorphize=on
#![feature(fn_traits, rustc_attrs, unboxed_closures)]

// This test checks that the polymorphization analysis considers a closure
// as using all generic parameters if it does an unsizing cast.

#[rustc_polymorphize_error]
fn foo2<T: Default>() {
    let _: T = Default::default();
    (|| {
        //~^ ERROR item has unused generic parameters
        let call: extern "rust-call" fn(_, _) = Fn::call;
        call(&|| {}, ());
        //~^ ERROR item has unused generic parameters
    })();
}

fn main() {
    foo2::<u32>();
}
