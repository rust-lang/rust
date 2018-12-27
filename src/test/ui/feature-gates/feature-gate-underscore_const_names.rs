#![feature(const_let)]

trait Trt {}
struct Str {}

impl Trt for Str {}

const _ : () = {
//~^ ERROR is unstable
    use std::marker::PhantomData;
    struct ImplementsTrait<T: Trt>(PhantomData<T>);
    let _ = ImplementsTrait::<Str>(PhantomData);
    ()
};

fn main() {}
