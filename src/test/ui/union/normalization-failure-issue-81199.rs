#[repr(C)]
union PtrRepr<T: ?Sized> {
    const_ptr: *const T,
    mut_ptr: *mut T,
    components: PtrComponents<T>,
    //~^ ERROR the trait bound `T: Pointee` is not satisfied in `PtrComponents<T>`
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
