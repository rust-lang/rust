//@ check-pass

pub struct Wrapper<T>(T);

pub trait Foo {
    fn bar() -> Wrapper<impl Sized>;
}

impl Foo for () {
    #[expect(refining_impl_trait)]
    fn bar() -> Wrapper<i32> {
        Wrapper(0)
    }
}

fn main() {}
