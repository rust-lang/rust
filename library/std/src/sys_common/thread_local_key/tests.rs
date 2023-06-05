use super::StaticKey;
use core::ptr;

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
