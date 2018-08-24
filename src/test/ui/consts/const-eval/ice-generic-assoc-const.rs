// compile-pass

pub trait Nullable {
    const NULL: Self;

    fn is_null(&self) -> bool;
}

impl<T> Nullable for *const T {
    const NULL: Self = 0 as *const T;

    fn is_null(&self) -> bool {
        *self == Self::NULL
    }
}

fn main() {
}
