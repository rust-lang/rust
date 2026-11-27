trait A<T> {}

trait B {
    type Type;
}

impl<T> B for T //~ ERROR overflow evaluating the requirement
where
    T: A<Self::Type>,
{
    type Type = bool;
}
fn main() {}
