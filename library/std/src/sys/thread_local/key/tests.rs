use super::{LazyKey, get, set};
use crate::ptr;

#[test]
fn smoke() {
    static K1: LazyKey = LazyKey::new(None);
    static K2: LazyKey = LazyKey::new(None);

    let k1 = K1.force();
    let k2 = K2.force();
    assert_ne!(k1, k2);

    assert_eq!(K1.force(), k1);
    assert_eq!(K2.force(), k2);

    unsafe {
        assert!(get(k1).is_null());
        assert!(get(k2).is_null());
        set(k1, ptr::without_provenance_mut(1));
        set(k2, ptr::without_provenance_mut(2));
        assert_eq!(get(k1) as usize, 1);
        assert_eq!(get(k2) as usize, 2);
    }
}

#[test]
fn destructors() {
    use crate::mem::ManuallyDrop;
    use crate::sync::Arc;
    use crate::thread;

    unsafe extern "C" fn destruct(ptr: *mut u8) {
        drop(unsafe { Arc::from_raw(ptr as *const ()) });
    }

    static KEY: LazyKey = LazyKey::new(Some(destruct));

    let shared1 = Arc::new(());
    let shared2 = Arc::clone(&shared1);

    let key = KEY.force();
    unsafe {
        assert!(get(key).is_null());
        set(key, Arc::into_raw(shared1) as *mut u8);
    }

    thread::spawn(move || unsafe {
        let key = KEY.force();
        assert!(get(key).is_null());
        set(key, Arc::into_raw(shared2) as *mut u8);
    })
    .join()
    .unwrap();

    // Leak the Arc, let the TLS destructor clean it up.
    let shared1 = unsafe { ManuallyDrop::new(Arc::from_raw(get(key) as *const ())) };
    assert_eq!(
        Arc::strong_count(&shared1),
        1,
        "destructor should have dropped the other reference on thread exit"
    );
}
