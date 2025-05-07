//@ check-pass
// this test ensures that UnsafePinned hides the niche of its inner type, just like UnsafeCell does

#![crate_type = "lib"]
#![feature(unsafe_pinned)]

use std::num::NonZero;
use std::pin::UnsafePinned;

macro_rules! assert_size_is {
    ($ty:ty = $size:expr) => {
        const _: () = assert!(size_of::<$ty>() == $size);
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

assert_size_is!(       UnsafePinned<       &()>   = size_of::<usize>());
assert_size_is!(       UnsafePinned<Option<&()>>  = size_of::<usize>());
assert_size_is!(Option<UnsafePinned<       &()>>  = size_of::<usize>() * 2);
assert_size_is!(Option<UnsafePinned<Option<&()>>> = size_of::<usize>() * 2);
