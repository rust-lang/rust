#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

#[repr(C)]
union PtrRepr<T> {
    const_ptr: *const T,
    mut_ptr: *mut T,
    components: PtrComponents<T>,
    //~^ ERROR the trait bound
    //~| ERROR field must implement `Copy`
}

#[repr(C)]
struct PtrComponents<T: Pointee> {
    data_pointer: *const (),
    metadata: <T as Pointee>::Metadata,
}



pub trait Pointee {
   type Metadata;
}

fn main() {}
