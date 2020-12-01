// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

pub const fn is_zst<T: ?Sized>() -> usize {
    if std::mem::size_of::<T>() == 0 {
        //[full]~^ ERROR the size for values of type `T` cannot be known at compilation time
        1
    } else {
        0
    }
}

pub struct AtLeastByte<T: ?Sized> {
    value: T,
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    pad: [u8; is_zst::<T>()],
    //[min]~^ ERROR generic parameters may not be used in const operations
    //[full]~^^ ERROR evaluation of constant value failed
}

fn main() {}
