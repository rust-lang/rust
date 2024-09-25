//@ check-pass

#![feature(return_type_notation)]

trait Trait {
    fn method() -> impl Sized;
}

struct Works;
impl Trait for Works {
    fn method() -> impl Sized {}
}

fn test<T: Trait>()
where
    T::method(..): Send,
{
}

fn main() {
    test::<Works>();
}
