use crate::ptr::NonNull;

#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "local_handle"]
pub struct LocalHandle<T: ?Sized> {
    ptr: NonNull<T>,
}

impl<T: ?Sized> Clone for LocalHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for LocalHandle<T> {}

impl<T: ?Sized> LocalHandle<T> {
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) } }
    }

    pub fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }
}
