#![warn(clippy::transmutes_expressible_as_ptr_casts)]

// rustc_typeck::check::cast contains documentation about when a cast `e as U` is 
// valid, which we quote from below.
use std::mem::transmute;

fn main() {
    // e is an integer and U is *U_0, while U_0: Sized; addr-ptr-cast
    let ptr_i32_transmute = unsafe {
        transmute::<isize, *const i32>(-1)
    };
    let ptr_i32 = -1isize as *const i32;

    // e has type *T, U is *U_0, and either U_0: Sized ...
    let ptr_i8_transmute = unsafe {
        transmute::<*const i32, *const i8>(ptr_i32)
    };
    let ptr_i8 = ptr_i32 as *const i8;

    let slice_ptr = &[0,1,2,3] as *const [i32];

    // ... or pointer_kind(T) = pointer_kind(U_0); ptr-ptr-cast
    let ptr_to_unsized_transmute = unsafe {
        transmute::<*const [i32], *const [u16]>(slice_ptr)
    };
    let ptr_to_unsized = slice_ptr as *const [u16];
    // TODO: We could try testing vtable casts here too, but maybe
    // we should wait until std::raw::TraitObject is stabilized?

    // e has type *T and U is a numeric type, while T: Sized; ptr-addr-cast
    let usize_from_int_ptr_transmute = unsafe {
        transmute::<*const i32, usize>(ptr_i32)
    };
    let usize_from_int_ptr = ptr_i32 as usize;

    let array_ref: &[i32; 4] = &[1,2,3,4];

    // e has type &[T; n] and U is *const T; array-ptr-cast
    let array_ptr_transmute = unsafe {
        transmute::<&[i32; 4], *const [i32; 4]>(array_ref)
    };
    let array_ptr = array_ref as *const [i32; 4];

    fn foo(_: usize) -> u8 { 42 }

    // e is a function pointer type and U has type *T, while T: Sized; fptr-ptr-cast
    let usize_ptr_transmute = unsafe {
        transmute::<fn(usize) -> u8, *const usize>(foo)
    };
    let usize_ptr_transmute = foo as *const usize;

    // e is a function pointer type and U is an integer; fptr-addr-cast
    let usize_from_fn_ptr_transmute = unsafe {
        transmute::<fn(usize) -> u8, usize>(foo)
    };
    let usize_from_fn_ptr = foo as *const usize;
}
