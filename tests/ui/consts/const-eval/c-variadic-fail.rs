//@ build-fail

#![feature(c_variadic)]
#![feature(c_variadic_const)]
#![feature(const_trait_impl)]
#![feature(const_destruct)]
#![feature(const_clone)]

use std::ffi::VaList;

const unsafe extern "C" fn read_n<const N: usize>(mut ap: ...) {
    let mut i = N;
    while i > 0 {
        i -= 1;
        let _ = ap.arg::<i32>();
    }
}

unsafe fn read_too_many() {
    // None passed, none read.
    const { read_n::<0>() }

    // One passed, none read. Ignoring arguments is fine.
    const { read_n::<0>(1) }

    // None passed, one read.
    const { read_n::<1>() }
    //~^ ERROR more C-variadic arguments read than were passed

    // One passed, two read.
    const { read_n::<2>(1) }
    //~^ ERROR more C-variadic arguments read than were passed
}

const unsafe extern "C" fn read_as<T: core::ffi::VaArgSafe>(mut ap: ...) -> T {
    ap.arg::<T>()
}

unsafe fn read_cast() {
    const { read_as::<i32>(1i32) };
    const { read_as::<u32>(1u32) };

    const { read_as::<i32>(1i32, 2u64, 3.0f64) };
    const { read_as::<u32>(1u32, 2u64, 3.0f64) };

    const { read_as::<i64>(1i64) };
    const { read_as::<u64>(1u64) };

    const { read_as::<u32>(1i32) };
    //~^ ERROR va_arg type mismatch: requested `u32`, but next argument is `i32`

    const { read_as::<i32>(1u32) };
    //~^ ERROR va_arg type mismatch: requested `i32`, but next argument is `u32`

    const { read_as::<i32>(1u64) };
    //~^ ERROR va_arg type mismatch: requested `i32`, but next argument is `u64`

    const { read_as::<f64>(1i32) };
    //~^ ERROR va_arg type mismatch: requested `f64`, but next argument is `i32`

    const { read_as::<*const u8>(1i32) };
    //~^ ERROR va_arg type mismatch: requested `*const u8`, but next argument is `i32`
}

fn use_after_free() {
    const unsafe extern "C" fn helper(ap: ...) -> [u8; size_of::<VaList>()] {
        unsafe { std::mem::transmute(ap) }
    }

    const {
        unsafe {
            let ap = helper(1, 2, 3);
            let mut ap = std::mem::transmute::<_, VaList>(ap);
            ap.arg::<i32>();
            //~^ ERROR memory access failed: ALLOC0 has been freed, so this pointer is dangling [E0080]
        }
    };
}

fn manual_copy() {
    const unsafe extern "C" fn helper(ap: ...) {
        // A copy created using Clone is valid, and can be used to read arguments.
        let mut copy = ap.clone();
        assert!(copy.arg::<i32>() == 1i32);

        let mut u = core::mem::MaybeUninit::uninit();
        unsafe { core::ptr::copy_nonoverlapping(&ap, u.as_mut_ptr(), 1) };

        // Manually creating the copy is fine.
        let mut copy = unsafe { u.assume_init() };

        // Using the copy is actually fine.
        let _ = copy.arg::<i32>();
        drop(copy);

        // But then using the original is UB.
        drop(ap);
    }

    const { unsafe { helper(1, 2, 3) } };
    //~^ ERROR va_end on unknown va_list allocation ALLOC0 [E0080]
}

fn main() {
    unsafe {
        read_too_many();
        read_cast();
        manual_copy();
    }
}
