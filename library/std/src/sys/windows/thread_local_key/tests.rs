// This file only tests the thread local key fallback.
// Windows targets with native thread local support do not use this.
#![cfg(not(target_thread_local))]

use super::StaticKey;
use crate::ptr;

#[test]
fn smoke() {
    static K1: StaticKey = StaticKey::new(None);
    static K2: StaticKey = StaticKey::new(None);

    unsafe {
        assert!(K1.get().is_null());
        assert!(K2.get().is_null());
        K1.set(ptr::invalid_mut(1));
        K2.set(ptr::invalid_mut(2));
        assert_eq!(K1.get() as usize, 1);
        assert_eq!(K2.get() as usize, 2);
    }
}

#[test]
fn destructors() {
    use crate::mem::ManuallyDrop;
    use crate::sync::Arc;
    use crate::thread;

    unsafe extern "C" fn destruct(ptr: *mut u8) {
        drop(Arc::from_raw(ptr as *const ()));
    }

    static KEY: StaticKey = StaticKey::new(Some(destruct));

    let shared1 = Arc::new(());
    let shared2 = Arc::clone(&shared1);

    unsafe {
        assert!(KEY.get().is_null());
        KEY.set(Arc::into_raw(shared1) as *mut u8);
    }

    thread::spawn(move || unsafe {
        assert!(KEY.get().is_null());
        KEY.set(Arc::into_raw(shared2) as *mut u8);
    })
    .join()
    .unwrap();

    // Leak the Arc, let the TLS destructor clean it up.
    let shared1 = unsafe { ManuallyDrop::new(Arc::from_raw(KEY.get() as *const ())) };
    assert_eq!(
        Arc::strong_count(&shared1),
        1,
        "destructor should have dropped the other reference on thread exit"
    );
}
