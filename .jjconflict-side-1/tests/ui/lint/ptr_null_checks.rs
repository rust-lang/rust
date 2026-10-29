//@ check-pass

use std::ptr;

extern "C" fn c_fn() {}
fn static_i32() -> &'static i32 { &1 }

fn main() {
    let fn_ptr = main;

    // ------------- Function pointers ---------------
    if (fn_ptr as *mut ()).is_null() {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as *const u8).is_null() {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as *const ()) == std::ptr::null() {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as *mut ()) == std::ptr::null_mut() {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as *const ()) == (0 as *const ()) {}
    //~^ WARN function pointers are not nullable
    if <*const _>::is_null(fn_ptr as *const ()) {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as *mut fn() as *const fn() as *const ()).is_null() {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as *mut fn() as *const fn()).cast_mut().is_null() {}
    //~^ WARN function pointers are not nullable
    if ((fn_ptr as *mut fn()).cast() as *const fn()).cast_mut().is_null() {}
    //~^ WARN function pointers are not nullable
    if (fn_ptr as fn() as *const ()).is_null() {}
    //~^ WARN function pointers are not nullable
    if (c_fn as *const fn()).is_null() {}
    //~^ WARN function pointers are not nullable

    // ---------------- References ------------------
    if (&mut 8 as *mut i32).is_null() {}
    //~^ WARN references are not nullable
    if ptr::from_mut(&mut 8).is_null() {}
    //~^ WARN call is never null
    if (&8 as *const i32).is_null() {}
    //~^ WARN references are not nullable
    if ptr::from_ref(&8).is_null() {}
    //~^ WARN call is never null
    if ptr::from_ref(&8).cast_mut().is_null() {}
    //~^ WARN call is never null
    if (ptr::from_ref(&8).cast_mut() as *mut i32).is_null() {}
    //~^ WARN call is never null
    if (&8 as *const i32) == std::ptr::null() {}
    //~^ WARN references are not nullable
    let ref_num = &8;
    if (ref_num as *const i32) == std::ptr::null() {}
    //~^ WARN references are not nullable
    if (b"\0" as *const u8).is_null() {}
    //~^ WARN references are not nullable
    if ("aa" as *const str).is_null() {}
    //~^ WARN references are not nullable
    if (&[1, 2] as *const i32).is_null() {}
    //~^ WARN references are not nullable
    if (&mut [1, 2] as *mut i32) == std::ptr::null_mut() {}
    //~^ WARN references are not nullable
    if (static_i32() as *const i32).is_null() {}
    //~^ WARN references are not nullable
    if (&*{ static_i32() } as *const i32).is_null() {}
    //~^ WARN references are not nullable

    // ---------------- Functions -------------------
    if ptr::NonNull::new(&mut 8).unwrap().as_ptr().is_null() {}
    //~^ WARN call is never null
    if ptr::NonNull::<u8>::dangling().as_ptr().is_null() {}
    //~^ WARN call is never null

    // ----------------------------------------------
    const ZPTR: *const () = 0 as *const _;
    const NOT_ZPTR: *const () = 1 as *const _;

    // unlike the uplifted clippy::fn_null_check lint we do
    // not lint on them
    if (fn_ptr as *const ()) == ZPTR {}
    if (fn_ptr as *const ()) == NOT_ZPTR {}
}
