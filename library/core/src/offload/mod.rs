// offload module
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::macros::builtin::offload_kernel;
use crate::marker::PhantomData;
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::offload;

// Region & Partitioning Strategy
#[unstable(feature = "offload", issue = "124509")]
pub unsafe trait PartitioningStrategy {
    type View<'a, T: 'a>;
    type ViewMut<'a, T: 'a>;

    fn index() -> usize;
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>>;
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>>;
}

#[derive(Copy, Clone)]
#[unstable(feature = "offload", issue = "124509")]
pub struct Region<'a, T, S: PartitioningStrategy> {
    ptr: *mut T,
    len: usize,
    _marker: core::marker::PhantomData<(&'a mut [T], S)>,
}

#[unstable(feature = "offload", issue = "124509")]
impl<'a, T, const N: usize, S> From<&PreloadMut<'a, [T; N]>> for Region<'a, T, S>
where
    S: PartitioningStrategy,
{
    fn from(p: &PreloadMut<'a, [T; N]>) -> Self {
        Self { ptr: p.cpu_ptr as *mut T, len: N, _marker: core::marker::PhantomData }
    }
}

struct RawRegion<'a, T> {
    pub ptr: *mut T,
    pub len: usize,
    _marker: core::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T> From<&'a mut [T]> for RawRegion<'a, T> {
    fn from(data: &'a mut [T]) -> Self {
        Self { ptr: data.as_mut_ptr(), len: data.len(), _marker: core::marker::PhantomData }
    }
}

impl<'a, T, const N: usize> From<&'a mut [T; N]> for RawRegion<'a, T> {
    fn from(data: &'a mut [T; N]) -> Self {
        Self { ptr: data.as_mut_ptr(), len: N, _marker: core::marker::PhantomData }
    }
}

#[unstable(feature = "offload", issue = "124509")]
impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    pub fn new<D>(data: D) -> Self
    where
        D: Into<RawRegion<'a, T>>,
    {
        let raw = data.into();
        Self { ptr: raw.ptr, len: raw.len, _marker: core::marker::PhantomData }
    }

    pub fn get(&self) -> Option<S::View<'_, T>> {
        unsafe { S::get(self.ptr as *const T, self.len) }
    }

    pub fn get_mut(&mut self) -> Option<S::ViewMut<'_, T>> {
        unsafe { S::get_mut(self.ptr, self.len) }
    }
}
