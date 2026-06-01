// offload module
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::macros::builtin::offload_kernel;
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::offload;

use crate::marker::PhantomData;

#[lang = "preload_type"]
#[unstable(feature = "offload", issue = "124509")]
pub struct Preload<'a, T: ?Sized> {
    cpu_ptr: *const T,
    _marker: PhantomData<&'a T>,
}

#[lang = "preload_mut_type"]
#[unstable(feature = "offload", issue = "124509")]
pub struct PreloadMut<'a, T: ?Sized> {
    cpu_ptr: *mut T,
    _marker: PhantomData<&'a mut T>,
}

#[lang = "preload"]
#[unstable(feature = "offload", issue = "124509")]
pub fn preload<'a, T: ?Sized>(x: &'a T) -> Preload<'a, T> {
    Preload { cpu_ptr: x as *const T, _marker: PhantomData }
}

#[lang = "preload_mut"]
#[unstable(feature = "offload", issue = "124509")]
pub fn preload_mut<'a, T: ?Sized>(x: &'a mut T) -> PreloadMut<'a, T> {
    PreloadMut { cpu_ptr: x as *mut T, _marker: PhantomData }
}

impl<T: ?Sized> Drop for PreloadMut<'_, T> {
    fn drop(&mut self) {
        // Intentionally empty.
        //
        // This exists so MIR creates Drop terminators for PreloadMut.
        // rustc codegen intercepts those terminators and emits the
        // offload return mapper.
    }
}

impl<T: ?Sized> Drop for Preload<'_, T> {
    fn drop(&mut self) {
        // Intentionally empty.
        //
        // This exists so MIR creates Drop terminators for Preload.
        // rustc codegen intercepts those terminators and emits the
        // offload return mapper.
    }
}
