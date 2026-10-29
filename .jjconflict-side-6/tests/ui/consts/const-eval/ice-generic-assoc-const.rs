//@ build-pass (tests post-monomorphisation failure)
#![crate_type = "lib"]

pub trait Nullable {
    const NULL: Self;

    fn is_null(&self) -> bool;
}

impl<T> Nullable for *const T {
    const NULL: Self = core::ptr::null::<T>();

    fn is_null(&self) -> bool {
        *self == Self::NULL
    }
}
