use crate::field::Subplace;
use crate::ops::place::{
    BorrowPlace, CreateHandle, CreateIndexHandle, DerefPlace, IndexPlace, MetadataPlace,
    PlaceHandle, PlaceProxy, ProjectPlace, ReadPlace, WritePlace,
};
use crate::ptr::Metadata;
use crate::{fmt, ptr};

impl<T: ?Sized> PlaceProxy for *mut T {
    type Target = T;
}

unsafe impl<T: ?Sized> CreateHandle for *mut T {
    type Handle = MutPtrHandle<T>;

    unsafe fn create_handle(ptr: *const Self) -> Self::Handle {
        MutPtrHandle(unsafe { ptr.read() })
    }
}

pub struct MutPtrHandle<T: ?Sized>(*mut T);

impl<T: ?Sized> Copy for MutPtrHandle<T> {}
impl<T: ?Sized> Clone for MutPtrHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> fmt::Debug for MutPtrHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LocalHandle").field(&self.0).finish()
    }
}

impl<T: ?Sized> PlaceHandle for MutPtrHandle<T> {
    type Target = T;
}

unsafe impl<T: ?Sized> MetadataPlace for MutPtrHandle<T> {
    unsafe fn metadata(self) -> Metadata<T> {
        ptr::metadata(self.0)
    }
}

unsafe impl<T> DerefPlace for MutPtrHandle<T>
where
    T: ?Sized + CreateHandle,
{
    type Derefed = T::Handle;

    unsafe fn deref_place(self) -> Self::Derefed {
        unsafe { T::create_handle(self.0) }
    }
}

unsafe impl<T, Idx> IndexPlace<Idx> for MutPtrHandle<T>
where
    T: ?Sized + CreateIndexHandle<Idx>,
{
    type Indexed = T::Handle;

    unsafe fn index_place(self, idx: Idx) -> Self::Indexed {
        unsafe { T::create_index_handle(self.0, idx) }
    }
}

unsafe impl<T, S> ProjectPlace<S> for MutPtrHandle<T>
where
    T: ?Sized,
    S: Subplace<Source = T>,
{
    type Projected = MutPtrHandle<S::Target>;

    unsafe fn project_place(self, subplace: S) -> Self::Projected {
        let (offset, meta) = subplace.offset(unsafe { self.metadata() });
        let data = unsafe { self.0.cast::<u8>().add(offset) };
        MutPtrHandle(ptr::from_raw_parts_mut(data, meta))
    }
}

unsafe impl<T> ReadPlace for MutPtrHandle<T> {
    unsafe fn read_place(self) -> Self::Target {
        unsafe { self.0.read() }
    }
}

unsafe impl<'a, T: ?Sized> BorrowPlace<'a, &'a T> for MutPtrHandle<T> {
    unsafe fn borrow_place(self) -> &'a T {
        unsafe { self.0.as_ref_unchecked() }
    }
}

unsafe impl<T> WritePlace for MutPtrHandle<T> {
    unsafe fn write_place(self, value: Self::Target) {
        unsafe { self.0.write(value) }
    }
}
