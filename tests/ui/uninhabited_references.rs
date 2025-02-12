#![warn(clippy::uninhabited_references)]
#![allow(clippy::missing_transmute_annotations)]
#![feature(never_type)]

fn ret_uninh_ref() -> &'static std::convert::Infallible {
    //~^ uninhabited_references
    unsafe { std::mem::transmute(&()) }
}

macro_rules! ret_something {
    ($name:ident, $ty:ty) => {
        fn $name(x: &$ty) -> &$ty {
            //~^ uninhabited_references
            &*x
            //~^ uninhabited_references
        }
    };
}

ret_something!(id_u32, u32);
ret_something!(id_never, !);

fn main() {
    let x = ret_uninh_ref();
    let _ = *x;
    //~^ uninhabited_references
}
