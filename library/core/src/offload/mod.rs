// offload module
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::macros::builtin::offload_kernel;
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::offload;

use crate::marker::PhantomData;

// We store a raw pointer instead of a reference, since the real location of the data will be on a
// GPU, at a different address. We only use the CPU pointer as a key to our runtime cpu-gpu pointer
// map. In the future we might even directly store the gpu ptr here, which would make it even
// clearer why we are using raw pointers instead of references.
// We still use a lifetime marker to prevent writes into the original cpu version of the object,
// while the data is on the gpu. Dropping this struct will inform the runtime that this pointer can
// no longer be used to access the gpu copy of the data. If the reference counter reaches zero, the
// runtime might delete the gpu copy of the preloaded value.
#[lang = "preload_type"]
#[unstable(feature = "offload", issue = "124509")]
pub struct Preload<'a, T: ?Sized> {
    pub cpu_ptr: *const T,
    _marker: PhantomData<&'a T>,
}

/// Waits until all previously submitted offload kernels on the current
/// host thread have completed.
///
/// This does not copy preloaded values back to the host and does not release
/// their device mappings. Those operations still occur when the corresponding
/// [`Preload`] or [`PreloadMut`] guard is dropped.
#[inline(always)]
#[unstable(feature = "offload", issue = "124509")]
pub fn offload_sync() {
    crate::intrinsics::offload_sync();
}

// We store a raw pointer instead of a reference, since the real location of the data will be on a
// GPU, at a different address. We only use the CPU pointer as a key to our runtime cpu-gpu pointer
// map. In the future we might even directly store the gpu ptr here, which would make it even
// clearer why we are using raw pointers instead of references.
// We still use a lifetime marker to prevent writes into the original cpu version of the object,
// while the data is on the gpu. Dropping this struct will force a copy of the data back from the
// gpu to the cpu, after which we can again safely use the original mutable reference.
#[lang = "preload_mut_type"]
#[unstable(feature = "offload", issue = "124509")]
pub struct PreloadMut<'a, T: ?Sized> {
    pub cpu_ptr: *mut T,
    _marker: PhantomData<&'a mut T>,
}

#[lang = "preload"]
#[unstable(feature = "offload", issue = "124509")]
pub fn preload<'a, T: ?Sized>(x: &'a T) -> Preload<'a, T> {
    let p = Preload { cpu_ptr: x as *const T, _marker: PhantomData };

    core::intrinsics::offload_preload(p.cpu_ptr, false);

    p
}

#[lang = "preload_mut"]
#[unstable(feature = "offload", issue = "124509")]
pub fn preload_mut<'a, T: ?Sized>(x: &'a mut T) -> PreloadMut<'a, T> {
    let p = PreloadMut { cpu_ptr: x as *mut T, _marker: PhantomData };

    core::intrinsics::offload_preload(p.cpu_ptr, true);

    p
}

impl<T: ?Sized> Drop for PreloadMut<'_, T> {
    fn drop(&mut self) {
        core::intrinsics::offload_preload_end(self.cpu_ptr, true);
    }
}

impl<T: ?Sized> Drop for Preload<'_, T> {
    fn drop(&mut self) {
        core::intrinsics::offload_preload_end(self.cpu_ptr, false);
    }
}
