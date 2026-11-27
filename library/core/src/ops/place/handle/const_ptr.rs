use crate::field::Subplace;
use crate::ops::place::{
    BorrowPlace, CreateHandle, CreateIndexHandle, DerefPlace, IndexPlace, MetadataPlace,
    PlaceHandle, PlaceProxy, ProjectPlace, ReadPlace,
};
use crate::ptr::Metadata;
use crate::{fmt, ptr};

impl<T: ?Sized> PlaceProxy for *const T {
    type Target = T;
}

unsafe impl<T: ?Sized> CreateHandle for *const T {
    type Handle = PtrHandle<T>;

    unsafe fn create_handle(ptr: *const Self) -> Self::Handle {
        PtrHandle(unsafe { ptr.read() })
    }
}

pub struct PtrHandle<T: ?Sized>(*const T);

impl<T: ?Sized> Copy for PtrHandle<T> {}
impl<T: ?Sized> Clone for PtrHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> fmt::Debug for PtrHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PtrHandle").field(&self.0).finish()
    }
}

impl<T: ?Sized> PlaceHandle for PtrHandle<T> {
    type Target = T;
}

unsafe impl<T: ?Sized> MetadataPlace for PtrHandle<T> {
    unsafe fn metadata(self) -> Metadata<T> {
        ptr::metadata(self.0)
    }
}

unsafe impl<T> DerefPlace for PtrHandle<T>
where
    T: ?Sized + CreateHandle,
{
    type Derefed = T::Handle;

    unsafe fn deref_place(self) -> Self::Derefed {
        unsafe { T::create_handle(self.0) }
    }
}

unsafe impl<T, Idx> IndexPlace<Idx> for PtrHandle<T>
where
    T: ?Sized + CreateIndexHandle<Idx>,
{
    type Indexed = T::Handle;

    unsafe fn index_place(self, idx: Idx) -> Self::Indexed {
        unsafe { T::create_index_handle(self.0, idx) }
    }
}

unsafe impl<T, S> ProjectPlace<S> for PtrHandle<T>
where
    T: ?Sized,
    S: Subplace<Source = T>,
{
    type Projected = PtrHandle<S::Target>;

    unsafe fn project_place(self, subplace: S) -> Self::Projected {
        let (offset, meta) = subplace.offset(unsafe { self.metadata() });
        let data = unsafe { self.0.cast::<u8>().add(offset) };
        PtrHandle(ptr::from_raw_parts(data, meta))
    }
}

unsafe impl<T> ReadPlace for PtrHandle<T> {
    unsafe fn read_place(self) -> Self::Target {
        unsafe { self.0.read() }
    }
}

unsafe impl<'a, T: ?Sized> BorrowPlace<'a, &'a T> for PtrHandle<T> {
    unsafe fn borrow_place(self) -> &'a T {
        unsafe { self.0.as_ref_unchecked() }
    }
}
