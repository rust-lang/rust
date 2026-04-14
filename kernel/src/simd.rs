use crate::BootRuntime;
use core::alloc::Layout;
use core::sync::atomic::{AtomicUsize, Ordering};

pub struct SimdGuard<'a, R: BootRuntime> {
    _rt: &'a R,
}

impl<'a, R: BootRuntime> SimdGuard<'a, R> {
    pub fn enter(rt: &'a R) -> Self {
        rt.simd_init_cpu();
        SimdGuard { _rt: rt }
    }
}

impl<'a, R: BootRuntime> Drop for SimdGuard<'a, R> {
    fn drop(&mut self) {}
}

pub fn with_simd<R: BootRuntime, T>(rt: &R, f: impl FnOnce() -> T) -> T {
    let _g = SimdGuard::enter(rt);
    f()
}

pub struct SimdState {
    buffer: [u8; 544],
    valid: bool,
}

unsafe impl Send for SimdState {}
unsafe impl Sync for SimdState {}

impl SimdState {
    pub fn new<R: BootRuntime>(_rt: &R) -> Self {
        Self {
            buffer: [0; 544],
            valid: false,
        }
    }

    fn aligned_ptr(&self) -> *mut u8 {
        let mut addr = self.buffer.as_ptr() as usize;
        let rem = addr % 16;
        if rem != 0 {
            addr = addr + (16 - rem);
        }
        addr as *mut u8
    }

    pub fn save<R: BootRuntime>(&mut self, rt: &R) {
        let ptr = self.aligned_ptr();
        if (ptr as usize) % 16 != 0 {
            crate::kinfo!(
                "SIMD ALIGNMENT ERROR! buffer is NOT 16-byte aligned! ptr={:p}",
                ptr
            );
        }
        unsafe { rt.simd_save(ptr) };
        self.valid = true;
    }

    pub fn restore<R: BootRuntime>(&self, rt: &R) {
        if self.valid {
            let ptr = self.aligned_ptr();
            if (ptr as usize) % 16 != 0 {
                crate::kinfo!(
                    "SIMD ALIGNMENT ERROR! buffer is NOT 16-byte aligned in restore! ptr={:p}",
                    ptr
                );
            }
            unsafe { rt.simd_restore(ptr) };
        }
    }
}

pub fn self_test<R: BootRuntime>(rt: &R) {
    use crate::kinfo;

    let (size, align) = rt.simd_state_layout();
    if size == 0 {
        kinfo!("SIMD self-test skipped (no SIMD support)");
        return;
    }

    with_simd(rt, || {
        #[repr(align(16))]
        struct AlignedStorage([u8; 1024]);

        let mut storage = AlignedStorage([0; 1024]);
        let buffer = storage.0.as_mut_ptr();

        if size > 1024 || align > 16 {
            kinfo!("SIMD self-test skipped (size/align too large for stack buffer)");
            return;
        }

        unsafe { rt.simd_save(buffer) };
        unsafe { rt.simd_restore(buffer) };

        kinfo!("SIMD self-test passed (save/restore cycle)");
    });
}
