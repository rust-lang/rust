#![feature(generic_const_items)]
#![allow(incomplete_features)]

// Ensure that we check if bounds on const items hold or not.

use std::convert::Infallible;

const C<T: Copy>: () = ();

const K<T>: () = ()
where
    Infallible: From<T>;

trait Trait<P> {
    const A: u32
    where
        P: Copy;

    const B<T>: u32
    where
        Infallible: From<T>;
}

impl<P> Trait<P> for () {
    const A: u32 = 0;
    const B<T>: u32 = 1;
}

fn main() {
    let () = C::<String>; //~ ERROR trait `Copy` is not implemented for `String`
    let () = K::<()>; //~ ERROR trait `From<()>` is not implemented for `Infallible`
    let _ = <() as Trait<Vec<u8>>>::A; //~ ERROR trait `Copy` is not implemented for `Vec<u8>`
    let _ = <() as Trait<&'static str>>::B::<()>; //~ ERROR trait `From<()>` is not implemented for `Infallible`
}
