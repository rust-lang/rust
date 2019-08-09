// run-pass
use std::ops::Deref;

struct ArenaSet<U: Deref, V=<U as Deref>::Target>(U, &'static V)
    where V: 'static + ?Sized;

static Z: [u8; 4] = [1,2,3,4];

fn arena() -> &'static ArenaSet<Vec<u8>> {
    fn __static_ref_initialize() -> ArenaSet<Vec<u8>> {
        ArenaSet(vec![], &Z)
    }
    unsafe {
        use std::sync::Once;
        fn require_sync<T: Sync>(_: &T) { }
        unsafe fn __stability() -> &'static ArenaSet<Vec<u8>> {
            use std::mem::transmute;
            static mut DATA: *const ArenaSet<Vec<u8>> = std::ptr::null_mut();

            static mut ONCE: Once = Once::new();
            ONCE.call_once(|| {
                DATA = transmute
                    ::<Box<ArenaSet<Vec<u8>>>, *const ArenaSet<Vec<u8>>>
                    (Box::new(__static_ref_initialize()));
            });

            &*DATA
        }
        let static_ref = __stability();
        require_sync(static_ref);
        static_ref
    }
}

fn main() {
    let &ArenaSet(ref u, v) = arena();
    assert!(u.is_empty());
    assert_eq!(v, Z);
}
