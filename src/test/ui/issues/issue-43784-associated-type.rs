pub trait Partial<X: ?Sized>: Copy {
}

pub trait Complete {
    type Assoc: Partial<Self>;
}

impl<T> Partial<T> for T::Assoc where
    T: Complete
{
}

impl<T> Complete for T { //~ ERROR the trait bound `T: std::marker::Copy` is not satisfied
    type Assoc = T;
}

fn main() {}
