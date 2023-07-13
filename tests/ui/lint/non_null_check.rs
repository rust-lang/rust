// check-pass

fn main() {
    let fn_ptr = main;

    if (fn_ptr as *mut ()).is_null() {}
    //~^ WARN can never be null, so checking
    if (fn_ptr as *const u8).is_null() {}
    //~^ WARN can never be null, so checking
    if (fn_ptr as *const ()) == std::ptr::null() {}
    //~^ WARN can never be null, so checking
    if (fn_ptr as *mut ()) == std::ptr::null_mut() {}
    //~^ WARN can never be null, so checking
    if (fn_ptr as *const ()) == (0 as *const ()) {}
    //~^ WARN can never be null, so checking
    if <*const _>::is_null(fn_ptr as *const ()) {}
    //~^ WARN can never be null, so checking
    if (fn_ptr as *mut fn() as *const fn() as *const ()).is_null() {}
    //~^ WARN can never be null, so checking
    if (fn_ptr as fn() as *const ()).is_null() {}
    //~^ WARN can never be null, so checking

    const ZPTR: *const () = 0 as *const _;
    const NOT_ZPTR: *const () = 1 as *const _;

    // unlike the uplifted clippy::fn_null_check lint we do
    // not lint on them
    if (fn_ptr as *const ()) == ZPTR {}
    if (fn_ptr as *const ()) == NOT_ZPTR {}

    // Non fn pointers

    let tup_ref: &_ = &(10u8, 10u8);
    if (tup_ref as *const (u8, u8)).is_null() {}
    //~^ WARN can never be null, so checking
    if (&mut (10u8, 10u8) as *mut (u8, u8)).is_null() {}
    //~^ WARN can never be null, so checking

    // We could warn on these too, but don't:
    if Box::into_raw(Box::new("hi")).is_null() {}

    let ptr = &mut () as *mut ();
    if core::ptr::NonNull::new(ptr).unwrap().as_ptr().is_null() {}

}
