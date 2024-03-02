pub trait Partial<X: ?Sized>: Copy {
}

pub trait Complete {
    type Assoc: Partial<Self>;
}

impl<T> Partial<T> for T::Assoc where
    T: Complete
{
}

impl<T> Complete for T {
    type Assoc = T; //~ ERROR trait `Copy` is not implemented for `T`
}

fn main() {}
