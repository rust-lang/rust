//@ build-fail

#![feature(c_variadic)]
#![feature(const_destruct)]
#![feature(c_variadic_const)]

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

fn main() {
    unsafe {
        read_too_many();
        read_cast();
    }
}
