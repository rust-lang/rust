#![feature(impl_trait_in_fn_trait_return)]

// A variant of `impl-fn-predefined-lifetimes.rs` which uses an explicit
// binder for the lifetime in the nested opaque type. This is intentionally
// not supported.

fn a<'a>() -> impl for<'b> Fn(&'a u8) -> (impl Sized + 'b) {
    //~^ ERROR: `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
    |x| x
}

fn _b<'a>() -> impl Fn(&'a u8) -> (impl Sized + 'a) {
    a()
}

fn main() {}
