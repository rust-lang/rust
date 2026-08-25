use std::sync::atomic::{AtomicPtr, Ordering};

#[inline(always)]
pub fn memrchr() {
    fn detect() {}

    static CROSS_CRATE_STATIC_ITEM: AtomicPtr<()> = AtomicPtr::new(detect as *mut ());

    unsafe {
        let fun = CROSS_CRATE_STATIC_ITEM.load(Ordering::SeqCst);
        std::mem::transmute::<*mut (), fn()>(fun)()
    }
}
