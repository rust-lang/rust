//@ known-bug: #110395
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, generic_const_exprs)]

#[const_trait]
trait ConstName {
    const NAME_BYTES: &'static [u8];
}

impl const ConstName for u8 {
    const NAME_BYTES: &'static [u8] = b"u8";
}

const fn name_len<T: ?Sized + ConstName>() -> usize {
    T::NAME_BYTES.len()
}

impl<T: ?Sized + ConstName> const ConstName for &T
where
    [(); name_len::<T>()]:,
{
    const NAME_BYTES: &'static [u8] = b"&T";
}

impl<T: ?Sized + ConstName> const ConstName for &mut T
where
    [(); name_len::<T>()]:,
{
    const NAME_BYTES: &'static [u8] = b"&mut T";
}

pub const ICE_1: &'static [u8] = <&&mut u8 as ConstName>::NAME_BYTES;
pub const ICE_2: &'static [u8] = <&mut &u8 as ConstName>::NAME_BYTES;

fn main() {}
