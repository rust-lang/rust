// edition:2018

pub trait Bar {
    type E: Copy;

    fn foo<T>() -> Self::E;
}

impl<S> Bar for S {
    type E = impl std::marker::Copy;
    fn foo<T>() -> Self::E {
        //~^ ERROR : Copy` is not satisfied [E0277]
        async {}
    }
}

fn main() {}
