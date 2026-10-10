// External crate for `opaque-hidden-ty-inference.rs`.
//
// `define` returns `WaddupGamers<T, closure>`. Its `Unpin` impl only holds if
// `<T as Leak>::Assoc` is that closure. The test's caller says `Assoc` is its
// own opaque type instead, so checking `Unpin` would try to set that opaque
// type to this closure.
pub struct WaddupGamers<T, U>(Option<T>, U);

impl<T: Leak<Assoc = U>, U> Unpin for WaddupGamers<T, U> {}

pub trait Leak {
    type Assoc;
}

impl<T> Leak for T {
    type Assoc = T;
}

pub fn define<T>() -> impl Sized {
    WaddupGamers(None::<T>, || ())
}
