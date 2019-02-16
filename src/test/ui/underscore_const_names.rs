// compile-pass

#![feature(underscore_const_names)]

trait Trt {}
struct Str {}
impl Trt for Str {}

macro_rules! check_impl {
    ($struct:ident,$trait:ident) => {
        const _ : () = {
            use std::marker::PhantomData;
            struct ImplementsTrait<T: $trait>(PhantomData<T>);
            let _ = ImplementsTrait::<$struct>(PhantomData);
            ()
        };
    }
}

#[deny(unused)]
const _ : () = ();

const _ : i32 = 42;
const _ : Str = Str{};

check_impl!(Str, Trt);
check_impl!(Str, Trt);

fn main() {
  check_impl!(Str, Trt);
  check_impl!(Str, Trt);
}
