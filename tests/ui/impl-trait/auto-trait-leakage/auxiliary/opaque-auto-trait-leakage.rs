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
