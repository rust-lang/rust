#![feature(generic_const_exprs)]

pub struct Error(());

pub trait FromSlice: Sized {
    const SIZE: usize = std::mem::size_of::<Self>();

    fn validate_slice(bytes: &[[u8; Self::SIZE]]) -> Result<(), Error>;
}
