//@ build-fail
//@ compile-flags: -Zextra-const-ub-checks
#![feature(c_variadic_va_arg_safe)]
#![feature(const_c_variadic)]
#![feature(const_trait_impl)]
#![feature(const_destruct)]
#![feature(const_clone)]

use std::ffi::{VaList, c_char, c_void, VaArgSafe};
use std::mem::{self, MaybeUninit};
use std::num::NonZero;
use std::ptr::{self, NonNull};

const unsafe extern "C" fn read_n<const N: usize>(mut ap: ...) {
    let mut i = N;
    while i > 0 {
        i -= 1;
        let _ = ap.next_arg::<i32>();
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

const unsafe extern "C" fn read_as<T: VaArgSafe>(mut ap: ...) -> T {
    ap.next_arg::<T>()
}

type ConcreteUsize = cfg_select! {
    target_pointer_width = "16" => u16,
    target_pointer_width = "32" => u32,
    target_pointer_width = "64" => u64,
};

type ConcreteIsize = cfg_select! {
    target_pointer_width = "16" => i16,
    target_pointer_width = "32" => i32,
    target_pointer_width = "64" => i64,
};

unsafe fn read_cast_numeric() {
    const { read_as::<i32>(1i32) };
    const { read_as::<u32>(1u32) };

    const { read_as::<i32>(1i32, 2u64, 3.0f64) };
    const { read_as::<u32>(1u32, 2u64, 3.0f64) };

    const { read_as::<i64>(1i64) };
    const { read_as::<u64>(1u64) };

    // A cast between signed and unsigned is OK so long as both types can represent the value.
    const { read_as::<u32>(1i32) };
    const { read_as::<i32>(1u32) };

    const { read_as::<ConcreteUsize>(1usize) };
    const { read_as::<usize>(1 as ConcreteUsize) };

    const { read_as::<ConcreteIsize>(-1isize) };
    const { read_as::<isize>(-1 as ConcreteIsize) };

    const { read_as::<u32>(-1i32) };
    //~^ ERROR va_arg value mismatch: value `-1_i32` cannot be represented by type `u32`
    const { read_as::<u32>(i32::MIN) };
    //~^ ERROR va_arg value mismatch: value `-2147483648_i32` cannot be represented by type `u32`
    const { read_as::<i32>(u32::MAX) };
    //~^ ERROR va_arg value mismatch: value `4294967295_u32` cannot be represented by type `i32`
    const { read_as::<i32>(i32::MAX as u32 + 1) };
    //~^ ERROR va_arg value mismatch: value `2147483648_u32` cannot be represented by type `i32`
    const { read_as::<i64>(u64::MAX) };
    //~^ ERROR va_arg value mismatch: value `18446744073709551615_u64` cannot be represented by type `i64`
    const { read_as::<i64>(i64::MAX as u64 + 1) };
    //~^ ERROR va_arg value mismatch: value `9223372036854775808_u64` cannot be represented by type `i64`

    const { read_as::<i32>(1u64) };
    //~^ ERROR va_arg type mismatch: requested `i32` is incompatible with next argument of type `u64`

    const { read_as::<f64>(1i32) };
    //~^ ERROR va_arg type mismatch: requested `f64` is incompatible with next argument of type `i32`
}

unsafe fn read_cast_pointer() {
    // A pointer mutability cast is OK.
    const { read_as::<*const i32>(ptr::dangling_mut::<i32>()) };
    const { read_as::<*mut i32>(ptr::dangling::<i32>()) };

    // A pointer cast is OK between compatible types.
    const { read_as::<*const i32>(ptr::dangling::<u32>()) };
    const { read_as::<*const i32>(ptr::dangling_mut::<u32>()) };
    const { read_as::<*mut i32>(ptr::dangling::<u32>()) };
    const { read_as::<*mut i32>(ptr::dangling_mut::<u32>()) };

    // Casting between pointers to i8/u8 and c_void is OK.
    const { read_as::<*const c_char>(ptr::dangling::<c_void>()) };
    const { read_as::<*const c_void>(ptr::dangling::<c_char>()) };
    const { read_as::<*const i8>(ptr::dangling::<c_void>()) };
    const { read_as::<*const c_void>(ptr::dangling::<i8>()) };
    const { read_as::<*const u8>(ptr::dangling::<c_void>()) };
    const { read_as::<*const c_void>(ptr::dangling::<u8>()) };

    const { read_as::<*const u16>(ptr::dangling::<c_void>()) };
    //~^ ERROR va_arg type mismatch: requested `*const u16` is incompatible with next argument of type `*const c_void`
    const { read_as::<*const c_void>(ptr::dangling::<u16>()) };
    //~^ ERROR va_arg type mismatch: requested `*const c_void` is incompatible with next argument of type `*const u16`
    const { read_as::<*const u16>(ptr::dangling::<i32>()) };
    //~^ ERROR va_arg type mismatch: requested `*const u16` is incompatible with next argument of type `*const i32`

    const { read_as::<*const u8>(1usize) };
    //~^ ERROR requested `*const u8` is incompatible with next argument of type `usize`
}

unsafe fn read_cast_null() {
    const { read_as::<NonNull<c_void>>(ptr::null::<c_void>()) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonNull<c_void>>(ptr::null_mut::<c_void>()) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonNull<c_void>>(None::<NonNull<c_void>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonNull<c_void>>(None::<NonNull<c_void>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1

    // Casting null to `None` is OK.
    const { read_as::<Option<NonNull<c_void>>>(ptr::null::<c_void>()) };
    const { read_as::<Option<NonNull<c_void>>>(ptr::null_mut::<c_void>()) };
    const { read_as::<*const c_void>(None::<NonNull<c_void>>) };
    const { read_as::<*mut c_void>(None::<NonNull<c_void>>) };
}

unsafe fn read_cast_zero() {
    const { read_as::<NonZero<u32>>(0u32) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<u32>>(0i32) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<i32>>(0u32) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<i32>>(0i32) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteUsize>>(0usize) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteUsize>>(0isize) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteIsize>>(0usize) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteIsize>>(0isize) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<u32>>(None::<NonZero<u32>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<u32>>(None::<NonZero<i32>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<i32>>(None::<NonZero<u32>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<i32>>(None::<NonZero<i32>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteUsize>>(None::<NonZero<usize>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteUsize>>(None::<NonZero<isize>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteIsize>>(None::<NonZero<usize>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1
    const { read_as::<NonZero<ConcreteIsize>>(None::<NonZero<isize>>) };
    //~^ ERROR encountered 0, but expected something greater or equal to 1

    // Casting zero to `None` is OK.
    const { read_as::<Option<NonZero<u32>>>(0u32) };
    const { read_as::<Option<NonZero<u32>>>(0i32) };
    const { read_as::<u32>(None::<NonZero<u32>>) };
    const { read_as::<i32>(None::<NonZero<u32>>) };
    const { read_as::<Option<NonZero<i32>>>(0u32) };
    const { read_as::<Option<NonZero<i32>>>(0i32) };
    const { read_as::<u32>(None::<NonZero<i32>>) };
    const { read_as::<i32>(None::<NonZero<i32>>) };
}

unsafe fn read_cast_lifetime() {
    // The types are equal up to free lifetimes.
    const { read_as::<*const &'static i32>(ptr::dangling::<&i32>()) };

    // Bound lifetimes do matter.
    const { read_as::<*const fn(&'static ())>(ptr::dangling::<for<'a> fn(&'a ())>()) };
    //~^ ERROR va_arg type mismatch: requested `*const fn(&())` is incompatible with next argument of type `*const for<'a> fn(&'a ())`
}

fn use_after_free() {
    const unsafe extern "C" fn helper(ap: ...) -> [u8; size_of::<VaList>()] {
        unsafe { mem::transmute(ap) }
    }

    const {
        unsafe {
            let ap = helper(1, 2, 3);
            let mut ap = mem::transmute::<_, VaList>(ap);
            ap.next_arg::<i32>();
            //~^ ERROR memory access failed: ALLOC$ID has been freed, so this pointer is dangling [E0080]
        }
    };
}

fn manual_copy_drop() {
    const unsafe extern "C" fn helper(ap: ...) {
        // A copy created using Clone is valid, and can be used to read arguments.
        let mut copy = ap.clone();
        assert!(copy.next_arg::<i32>() == 1i32);

        let mut copy: VaList = unsafe { mem::transmute_copy(&ap) };

        // Using the copy is actually fine.
        let _ = copy.next_arg::<i32>();
        drop(copy);

        // But then using the original is UB.
        drop(ap);
    }

    const { unsafe { helper(1, 2, 3) } };
    //~^ ERROR using ALLOC$ID as variable argument list pointer but it does not point to a variable argument list [E0080]
}

fn manual_copy_forget() {
    const unsafe extern "C" fn helper(ap: ...) {
        let mut copy: VaList = unsafe { mem::transmute_copy(&ap) };

        // Using the copy is actually fine.
        let _ = copy.next_arg::<i32>();
        mem::forget(copy);

        // The read (via `copy`) deallocated the original allocation.
        drop(ap);
    }

    const { unsafe { helper(1, 2, 3) } };
    //~^ ERROR using ALLOC$ID as variable argument list pointer but it does not point to a variable argument list [E0080]
}

fn manual_copy_read() {
    const unsafe extern "C" fn helper(mut ap: ...) {
        let mut copy: VaList = unsafe { mem::transmute_copy(&ap) };

        // Reading from `ap` after reading from `copy` is UB.
        let _ = copy.next_arg::<i32>();
        let _ = ap.next_arg::<i32>();
    }

    const { unsafe { helper(1, 2, 3) } };
    //~^ ERROR using ALLOC$ID as variable argument list pointer but it does not point to a variable argument list [E0080]
}

fn drop_of_invalid() {
    const {
        let mut invalid: MaybeUninit<VaList> = MaybeUninit::zeroed();
        let ap = unsafe { invalid.assume_init() };
    }
    //~^ ERROR pointer not dereferenceable: pointer must point to some allocation, but got null pointer [E0080]
}

fn main() {
    unsafe {
        read_too_many();
        read_cast_numeric();
        read_cast_pointer();
        read_cast_lifetime();
        manual_copy_read();
        manual_copy_drop();
        manual_copy_forget();
        drop_of_invalid();
    }
}
