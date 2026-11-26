use crate::field::Subplace;
use crate::marker::PhantomData;
use crate::ops::place::{
    BorrowPlace, CreateHandle, CreateIndexHandle, DerefPlace, IndexPlace, MetadataPlace,
    PlaceHandle, PlaceProxy, ProjectPlace, ReadPlace, WritePlace,
};
use crate::ptr::Metadata;
use crate::{fmt, ptr};

impl<'a, T: ?Sized> PlaceProxy for &'a mut T {
    type Target = T;
}

unsafe impl<'a, T: ?Sized> CreateHandle for &'a mut T {
    type Handle = MutHandle<'a, T>;

    unsafe fn create_handle(ptr: *const Self) -> Self::Handle {
        let ptr = ptr.cast::<*mut T>();
        MutHandle { ptr: unsafe { ptr.read() }, _lt: PhantomData }
    }
}

pub struct MutHandle<'a, T: ?Sized> {
    ptr: *mut T,
    _lt: PhantomData<&'a mut T>,
}

impl<'a, T: ?Sized> Copy for MutHandle<'a, T> {}
impl<'a, T: ?Sized> Clone for MutHandle<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: ?Sized> fmt::Debug for MutHandle<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MutHandle").field(&self.ptr).finish()
    }
}

impl<'a, T: ?Sized> PlaceHandle for MutHandle<'a, T> {
    type Target = T;
}

unsafe impl<'a, T: ?Sized> MetadataPlace for MutHandle<'a, T> {
    unsafe fn metadata(self) -> Metadata<T> {
        ptr::metadata(self.ptr)
    }
}

unsafe impl<'a, T> DerefPlace for MutHandle<'a, T>
where
    T: ?Sized + CreateHandle,
{
    type Derefed = T::Handle;

    unsafe fn deref_place(self) -> Self::Derefed {
        unsafe { T::create_handle(self.ptr) }
    }
}

unsafe impl<'a, T, Idx> IndexPlace<Idx> for MutHandle<'a, T>
where
    T: ?Sized + CreateIndexHandle<Idx>,
{
    type Indexed = T::Handle;

    unsafe fn index_place(self, idx: Idx) -> Self::Indexed {
        unsafe { T::create_index_handle(self.ptr, idx) }
    }
}

unsafe impl<'a, T, S> ProjectPlace<S> for MutHandle<'a, T>
where
    T: ?Sized,
    S: Subplace<Source = T>,
    S::Target: 'a,
{
    type Projected = MutHandle<'a, S::Target>;

    unsafe fn project_place(self, subplace: S) -> Self::Projected {
        let (offset, meta) = subplace.offset(unsafe { self.metadata() });
        let data = unsafe { self.ptr.cast::<u8>().add(offset) };
        MutHandle { ptr: ptr::from_raw_parts_mut(data, meta), _lt: PhantomData }
    }
}

unsafe impl<'a, T> ReadPlace for MutHandle<'a, T> {
    unsafe fn read_place(self) -> Self::Target {
        unsafe { self.ptr.read() }
    }
}

unsafe impl<'a, T> WritePlace for MutHandle<'a, T> {
    unsafe fn write_place(self, value: Self::Target) {
        unsafe { self.ptr.write(value) }
    }
}

unsafe impl<'a, T: ?Sized> BorrowPlace<'a, &'a T> for MutHandle<'a, T> {
    unsafe fn borrow_place(self) -> &'a T {
        unsafe { self.ptr.as_ref_unchecked() }
    }
}

unsafe impl<'a, T: ?Sized> BorrowPlace<'a, &'a mut T> for MutHandle<'a, T> {
    unsafe fn borrow_place(self) -> &'a mut T {
        unsafe { self.ptr.as_mut_unchecked() }
    }
}
