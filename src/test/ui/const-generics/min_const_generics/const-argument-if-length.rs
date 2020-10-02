#![feature(min_const_generics)]

pub const fn is_zst<T: ?Sized>() -> usize {
    if std::mem::size_of::<T>() == 0 {
        1
    } else {
        0
    }
}

pub struct AtLeastByte<T: ?Sized> {
    value: T,
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    pad: [u8; is_zst::<T>()],
    //~^ ERROR generic parameters must not be used inside of non-trivial constant values
}

fn main() {}
