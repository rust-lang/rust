// This is just `mbe-async-trait-bound-theoretical-regression.rs` in practice.

//@ edition:2021
// for the `impl` + keyword test

macro_rules! impl_primitive {
    ($ty:ty) => {
        compile_error!("whoops");
    };
    (impl async) => {};
}

impl_primitive!(impl async);
//~^ ERROR expected identifier, found `<eof>`
//~| ERROR `async` trait bounds are unstable

fn main() {}
