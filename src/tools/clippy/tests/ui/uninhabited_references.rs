#![warn(clippy::uninhabited_references)]
#![feature(never_type)]

fn ret_uninh_ref() -> &'static std::convert::Infallible {
    unsafe { std::mem::transmute(&()) }
}

macro_rules! ret_something {
    ($name:ident, $ty:ty) => {
        fn $name(x: &$ty) -> &$ty {
            &*x
        }
    };
}

ret_something!(id_u32, u32);
ret_something!(id_never, !);

fn main() {
    let x = ret_uninh_ref();
    let _ = *x;
}
