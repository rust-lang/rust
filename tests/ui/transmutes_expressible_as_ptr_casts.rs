// run-rustfix
#![warn(clippy::transmutes_expressible_as_ptr_casts)]
// These two warnings currrently cover the cases transmutes_expressible_as_ptr_casts
// would otherwise be responsible for
#![warn(clippy::useless_transmute)]
#![warn(clippy::transmute_ptr_to_ptr)]
#![allow(dead_code, unused_unsafe, clippy::borrow_as_ptr)]

use std::mem::{size_of, transmute};

// rustc_typeck::check::cast contains documentation about when a cast `e as U` is
// valid, which we quote from below.
fn main() {
    // We should see an error message for each transmute, and no error messages for
    // the casts, since the casts are the recommended fixes.

    // e is an integer and U is *U_0, while U_0: Sized; addr-ptr-cast
    let _ptr_i32_transmute = unsafe { transmute::<usize, *const i32>(usize::MAX) };
    let ptr_i32 = usize::MAX as *const i32;

    // e has type *T, U is *U_0, and either U_0: Sized ...
    let _ptr_i8_transmute = unsafe { transmute::<*const i32, *const i8>(ptr_i32) };
    let _ptr_i8 = ptr_i32 as *const i8;

    let slice_ptr = &[0, 1, 2, 3] as *const [i32];

    // ... or pointer_kind(T) = pointer_kind(U_0); ptr-ptr-cast
    let _ptr_to_unsized_transmute = unsafe { transmute::<*const [i32], *const [u16]>(slice_ptr) };
    let _ptr_to_unsized = slice_ptr as *const [u16];
    // TODO: We could try testing vtable casts here too, but maybe
    // we should wait until std::raw::TraitObject is stabilized?

    // e has type *T and U is a numeric type, while T: Sized; ptr-addr-cast
    let _usize_from_int_ptr_transmute = unsafe { transmute::<*const i32, usize>(ptr_i32) };
    let _usize_from_int_ptr = ptr_i32 as usize;

    let array_ref: &[i32; 4] = &[1, 2, 3, 4];

    // e has type &[T; n] and U is *const T; array-ptr-cast
    let _array_ptr_transmute = unsafe { transmute::<&[i32; 4], *const [i32; 4]>(array_ref) };
    let _array_ptr = array_ref as *const [i32; 4];

    fn foo(_: usize) -> u8 {
        42
    }

    // e is a function pointer type and U has type *T, while T: Sized; fptr-ptr-cast
    let _usize_ptr_transmute = unsafe { transmute::<fn(usize) -> u8, *const usize>(foo) };
    let _usize_ptr_transmute = foo as *const usize;

    // e is a function pointer type and U is an integer; fptr-addr-cast
    let _usize_from_fn_ptr_transmute = unsafe { transmute::<fn(usize) -> u8, usize>(foo) };
    let _usize_from_fn_ptr = foo as *const usize;
}

// If a ref-to-ptr cast of this form where the pointer type points to a type other
// than the referenced type, calling `CastCheck::do_check` has been observed to
// cause an ICE error message. `do_check` is currently called inside the
// `transmutes_expressible_as_ptr_casts` check, but other, more specific lints
// currently prevent it from being called in these cases. This test is meant to
// fail if the ordering of the checks ever changes enough to cause these cases to
// fall through into `do_check`.
fn trigger_do_check_to_emit_error(in_param: &[i32; 1]) -> *const u8 {
    unsafe { transmute::<&[i32; 1], *const u8>(in_param) }
}

#[repr(C)]
struct Single(u64);

#[repr(C)]
struct Pair(u32, u32);

fn cannot_be_expressed_as_pointer_cast(in_param: Single) -> Pair {
    assert_eq!(size_of::<Single>(), size_of::<Pair>());

    unsafe { transmute::<Single, Pair>(in_param) }
}
