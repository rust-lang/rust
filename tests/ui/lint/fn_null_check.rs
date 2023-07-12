// check-pass

fn main() {
    let fn_ptr = main;

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
    if (fn_ptr as fn() as *const ()).is_null() {}
    //~^ WARN function pointers are not nullable

    const ZPTR: *const () = 0 as *const _;
    const NOT_ZPTR: *const () = 1 as *const _;

    // unlike the uplifted clippy::fn_null_check lint we do
    // not lint on them
    if (fn_ptr as *const ()) == ZPTR {}
    if (fn_ptr as *const ()) == NOT_ZPTR {}
}
