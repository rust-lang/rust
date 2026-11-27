use crate::field::Subplace;
use crate::ops::place::{
    BorrowPlace, CreateHandle, CreateIndexHandle, DerefPlace, DropPlace, IndexPlace, MetadataPlace,
    MovePlace, PlaceHandle, ProjectPlace, ReadPlace, WritePlace,
};
use crate::ptr::Metadata;
use crate::{fmt, ptr};

pub struct LocalHandle<T: ?Sized>(*mut T);

impl<T: ?Sized> Copy for LocalHandle<T> {}
impl<T: ?Sized> Clone for LocalHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> fmt::Debug for LocalHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LocalHandle").field(&self.get()).finish()
    }
}

impl<T: ?Sized> LocalHandle<T> {
    pub unsafe fn new(ptr: *const T) -> Self {
        Self(ptr.cast_mut())
    }

    pub fn get(self) -> *mut T {
        self.0
    }
}

impl<T: ?Sized> PlaceHandle for LocalHandle<T> {
    type Target = T;
}

unsafe impl<T: ?Sized> MetadataPlace for LocalHandle<T> {
    unsafe fn metadata(self) -> Metadata<T> {
        ptr::metadata(self.get())
    }
}

unsafe impl<T> DerefPlace for LocalHandle<T>
where
    T: ?Sized + CreateHandle,
{
    type Derefed = T::Handle;

    unsafe fn deref_place(self) -> Self::Derefed {
        unsafe { T::create_handle(self.get()) }
    }
}

unsafe impl<T, Idx> IndexPlace<Idx> for LocalHandle<T>
where
    T: ?Sized + CreateIndexHandle<Idx>,
{
    type Indexed = T::Handle;

    unsafe fn index_place(self, idx: Idx) -> Self::Indexed {
        unsafe { T::create_index_handle(self.get(), idx) }
    }
}

unsafe impl<T, S> ProjectPlace<S> for LocalHandle<T>
where
    T: ?Sized,
    S: Subplace<Source = T>,
{
    type Projected = LocalHandle<S::Target>;

    unsafe fn project_place(self, subplace: S) -> Self::Projected {
        let (offset, meta) = subplace.offset(unsafe { self.metadata() });
        let data = unsafe { self.get().cast::<u8>().add(offset) };
        unsafe { LocalHandle::new(ptr::from_raw_parts_mut(data, meta)) }
    }
}

unsafe impl<T> ReadPlace for LocalHandle<T> {
    unsafe fn read_place(self) -> Self::Target {
        unsafe { self.0.read() }
    }
}

unsafe impl<T> MovePlace for LocalHandle<T> {}

unsafe impl<T> WritePlace for LocalHandle<T> {
    unsafe fn write_place(self, value: Self::Target) {
        unsafe { self.0.write(value) }
    }
}

unsafe impl<'a, T: ?Sized> BorrowPlace<'a, &'a mut T> for LocalHandle<T> {
    unsafe fn borrow_place(self) -> &'a mut T {
        unsafe { self.0.as_mut_unchecked() }
    }
}

unsafe impl<'a, T: ?Sized> BorrowPlace<'a, &'a T> for LocalHandle<T> {
    unsafe fn borrow_place(self) -> &'a T {
        unsafe { self.0.as_ref_unchecked() }
    }
}

unsafe impl<T: ?Sized> DropPlace for LocalHandle<T> {
    unsafe fn drop_place(self) {
        unsafe { self.get().drop_in_place() }
    }
}
