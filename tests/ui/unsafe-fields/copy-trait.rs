//@ compile-flags: --crate-type=lib

#![feature(unsafe_fields)]
#![allow(incomplete_features)]
#![deny(missing_copy_implementations)]

mod good_safe_impl {
    enum SafeEnum {
        Safe(u8),
    }

    impl Copy for SafeEnum {}
}

mod bad_safe_impl {
    enum UnsafeEnum {
        Safe(u8),
        Unsafe { unsafe field: u8 },
    }

    impl Copy for UnsafeEnum {}
    //~^ ERROR the trait `Copy` requires an `unsafe impl` declaration
}

mod good_unsafe_impl {
    enum UnsafeEnum {
        Safe(u8),
        Unsafe { unsafe field: u8 },
    }

    unsafe impl Copy for UnsafeEnum {}
}

mod bad_unsafe_impl {
    enum SafeEnum {
        Safe(u8),
    }

    unsafe impl Copy for SafeEnum {}
    //~^ ERROR implementing the trait `Copy` is not unsafe
}
