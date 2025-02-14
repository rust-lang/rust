//@ build-pass

#![crate_type = "lib"]
#![feature(unsafe_pinned)]

use std::num::NonZero;
use std::pin::UnsafePinned;

struct Size<const N: usize>;

const USIZE_SIZE: usize = size_of::<usize>();

macro_rules! assert_size_is {
    ($ty:ty = $size:expr) => {
        const _: Size<{ $size }> = Size::<{ size_of::<$ty>() }>;
    };
}

assert_size_is!(UnsafePinned<()> = 0);
assert_size_is!(UnsafePinned<u8> = 1);

assert_size_is!(       UnsafePinned<               u32>    = 4);
assert_size_is!(       UnsafePinned<       NonZero<u32>>   = 4);
assert_size_is!(       UnsafePinned<Option<NonZero<u32>>>  = 4);
assert_size_is!(Option<UnsafePinned<               u32>>   = 8);
assert_size_is!(Option<UnsafePinned<       NonZero<u32>>>  = 8);
assert_size_is!(Option<UnsafePinned<Option<NonZero<u32>>>> = 8);

assert_size_is!(       UnsafePinned<       &()>   = USIZE_SIZE);
assert_size_is!(       UnsafePinned<Option<&()>>  = USIZE_SIZE);
assert_size_is!(Option<UnsafePinned<       &()>>  = USIZE_SIZE * 2);
assert_size_is!(Option<UnsafePinned<Option<&()>>> = USIZE_SIZE * 2);
