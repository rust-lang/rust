// run-pass
#![feature(const_int_nonzero)]

use std::num::{
    NonZeroI8,
    NonZeroU8,
    NonZeroI32,
    NonZeroUsize,
};

macro_rules! assert_same_const {
    ($(const $ident:ident: $ty:ty = $exp:expr;)+) => {
        $(const $ident: $ty = $exp;)+

        pub fn main() {
            $(assert_eq!($exp, $ident);)+
        }
    }
}

assert_same_const! {
    const NON_ZERO_NEW_1: Option<NonZeroI8> = NonZeroI8::new(1);
    const NON_ZERO_NEW_2: Option<NonZeroI8> = NonZeroI8::new(0);
    const NON_ZERO_NEW_3: Option<NonZeroI8> = NonZeroI8::new(-38);
    const NON_ZERO_NEW_4: Option<NonZeroU8> = NonZeroU8::new(1);
    const NON_ZERO_NEW_5: Option<NonZeroU8> = NonZeroU8::new(0);
    const NON_ZERO_NEW_6: Option<NonZeroI32> = NonZeroI32::new(1);
    const NON_ZERO_NEW_7: Option<NonZeroI32> = NonZeroI32::new(0);
    const NON_ZERO_NEW_8: Option<NonZeroI32> = NonZeroI32::new(-38);
    const NON_ZERO_NEW_9: Option<NonZeroUsize> = NonZeroUsize::new(1);
    const NON_ZERO_NEW_10: Option<NonZeroUsize> = NonZeroUsize::new(0);

    // Option::unwrap isn't supported in const yet, so we use new_unchecked.
    const NON_ZERO_GET_1: i8 = unsafe { NonZeroI8::new_unchecked(1) }.get();
    const NON_ZERO_GET_2: i8 = unsafe { NonZeroI8::new_unchecked(-38) }.get();
    const NON_ZERO_GET_3: u8 = unsafe { NonZeroU8::new_unchecked(1) }.get();
    const NON_ZERO_GET_4: i32 = unsafe { NonZeroI32::new_unchecked(1) }.get();
    const NON_ZERO_GET_5: i32 = unsafe { NonZeroI32::new_unchecked(-38) }.get();
    const NON_ZERO_GET_6: usize = unsafe { NonZeroUsize::new_unchecked(1) }.get();

    const NON_ZERO_NEW_UNCHECKED_1: NonZeroI8 = unsafe { NonZeroI8::new_unchecked(1) };
    const NON_ZERO_NEW_UNCHECKED_2: NonZeroI8 = unsafe { NonZeroI8::new_unchecked(-38) };
    const NON_ZERO_NEW_UNCHECKED_3: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(1) };
    const NON_ZERO_NEW_UNCHECKED_4: NonZeroI32 = unsafe { NonZeroI32::new_unchecked(1) };
    const NON_ZERO_NEW_UNCHECKED_5: NonZeroI32 = unsafe { NonZeroI32::new_unchecked(-38) };
    const NON_ZERO_NEW_UNCHECKED_6: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1) };
}
