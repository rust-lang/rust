// build-pass (FIXME(62277): could be check-pass?)

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

fn main() {
}
