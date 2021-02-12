// run-rustfix
#[allow(dead_code)]
#[repr(C)]
union PtrRepr<T: ?Sized> {
    const_ptr: *const T,
    mut_ptr: *mut T,
    components: std::mem::ManuallyDrop<PtrComponents<T>>,
    //~^ ERROR the trait bound `T: Pointee` is not satisfied
}

#[repr(C)]
struct PtrComponents<T: Pointee + ?Sized> {
    data_address: *const (),
    metadata: <T as Pointee>::Metadata,
}

pub trait Pointee {
   type Metadata;
}

fn main() {}
