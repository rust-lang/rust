use super::{Key, StaticKey};
use core::ptr;

fn assert_sync<T: Sync>() {}
fn assert_send<T: Send>() {}

#[test]
fn smoke() {
    assert_sync::<Key>();
    assert_send::<Key>();

    let k1 = Key::new(None);
    let k2 = Key::new(None);
    assert!(k1.get().is_null());
    assert!(k2.get().is_null());
    k1.set(ptr::invalid_mut(1));
    k2.set(ptr::invalid_mut(2));
    assert_eq!(k1.get() as usize, 1);
    assert_eq!(k2.get() as usize, 2);
}

#[test]
fn statik() {
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
