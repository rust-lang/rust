#[repr(C)]
union PtrRepr<T: ?Sized> {
    const_ptr: *const T,
    mut_ptr: *mut T,
    components: PtrComponents<T>,
    //~^ ERROR the trait bound
    //~| ERROR field must implement `Copy`
}

#[repr(C)]
struct PtrComponents<T: Pointee + ?Sized> {
    data_pointer: *const (),
    metadata: <T as Pointee>::Metadata,
}



pub trait Pointee {
   type Metadata;
}

fn main() {}
