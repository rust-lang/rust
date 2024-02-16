//@ build-pass (FIXME(62277): could be check-pass?)

#![deny(unused)]

pub trait Trt {}
pub struct Str {}
impl Trt for Str {}

macro_rules! check_impl {
    ($struct:ident, $trait:ident) => {
        const _ : () = {
            use std::marker::PhantomData;
            struct ImplementsTrait<T: $trait>(PhantomData<T>);
            let _ = ImplementsTrait::<$struct>(PhantomData);
            ()
        };
    }
}

const _ : () = ();

const _ : i32 = 42;
const _ : Str = Str{};

check_impl!(Str, Trt);
check_impl!(Str, Trt);

fn main() {
  check_impl!(Str, Trt);
  check_impl!(Str, Trt);
}
