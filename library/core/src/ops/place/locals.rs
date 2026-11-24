pub struct LocalHandle<T: ?Sized>(*mut T);

impl<T: ?Sized> Copy for LocalHandle<T> {}
impl<T: ?Sized> Clone for LocalHandle<T> {
    fn clone(&self) -> Self {
        *self
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
    T: ?Sized + IndexProxy<Idx>,
{
    type Derefed = T::Handle;

    unsafe fn deref_place(self) -> Self::Derefed {
        compiler_error!("TODO: create a index version of CreateHandle?");
    }
}

unsafe impl<T, S> ProjectPlace<S> for LocalHandle<T>
where
    S: Subplace<Source = T, Target: Sized>,
{
    type Projected = LocalHandle<S::Target>;

    unsafe fn project_place(self, subplace: S) -> Self::Projected {
        unsafe { LocalHandle::new(self.get().byte_add(subplace.offset()).cast()) }
    }
}

impl<T: ?Sized> ReadPlace for LocalHandle<T> {
    unsafe fn read_place(self) -> Self::Target {
        unsafe { self.0.read() }
    }
}

impl<T: ?Sized> MovePlace for LocalHandle<T> {}

impl<T> WritePlace for LocalHandle<T> {
    unsafe fn write_place(self, value: Self::Target) {
        unsafe { self.0.write(value) }
    }
}

impl<'a, T: 'a + ?Sized> BorrowPlace<'a, &'a mut T> for LocalHandle<T> {
    unsafe fn borrow_place(self) -> &'a mut T {
        unsafe { self.0.as_mut() }
    }
}

impl<'a, T: 'a + ?Sized> BorrowPlace<'a, &'a T> for LocalHandle<T> {
    unsafe fn borrow_place(self) -> &'a T {
        unsafe { self.0.as_ref() }
    }
}

impl<T: ?Sized> MetadataPlace for LocalHandle<T> {
    unsafe fn metadata(self) -> <Self::Target as Pointee>::Metadata {
        ptr::metadata(self.get())
    }
}

unsafe impl<T: ?Sized> DropPlace for LocalHandle<T> {
    unsafe fn drop_place(self) {
        unsafe { self.get().drop_in_place() }
    }
}
