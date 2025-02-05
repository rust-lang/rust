use crate::mem;

pub unsafe fn at_process_exit(cb: unsafe extern "C" fn()) {
    // Miri does not support atexit.
    #[cfg(not(miri))]
    assert_eq!(unsafe { libc::atexit(mem::transmute(cb)) }, 0);

    #[cfg(miri)]
    let _ = cb;
}
