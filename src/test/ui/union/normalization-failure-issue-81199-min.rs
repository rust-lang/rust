union PtrRepr<T: ?Sized> {
    const_ptr: *const T,
    mut_ptr: *mut T,
    components: <T as Pointee>::Metadata
    //~^ ERROR the trait bound `T: Pointee` is not satisfied
}

pub trait Pointee {
   type Metadata;
}

fn main() {}
