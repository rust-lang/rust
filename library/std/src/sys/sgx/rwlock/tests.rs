use super::*;
use crate::ptr;

// Verify that the byte pattern libunwind uses to initialize an RwLock is
// equivalent to the value of RwLock::new(). If the value changes,
// `src/UnwindRustSgx.h` in libunwind needs to be changed too.
#[test]
fn test_c_rwlock_initializer() {
    const C_RWLOCK_INIT: *mut () = ptr::null_mut();

    // For the test to work, we need the padding/unused bytes in RwLock to be
    // initialized as 0. In practice, this is the case with statics.
    static RUST_RWLOCK_INIT: RwLock = RwLock::new();

    unsafe {
        // If the assertion fails, that not necessarily an issue with the value
        // of C_RWLOCK_INIT. It might just be an issue with the way padding
        // bytes are initialized in the test code.
        assert_eq!(crate::mem::transmute_copy::<_, *mut ()>(&RUST_RWLOCK_INIT), C_RWLOCK_INIT);
    };
}
