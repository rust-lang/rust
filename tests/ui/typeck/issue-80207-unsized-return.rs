//@ check-pass

trait Foo {
    fn do_stuff() -> Self;
}

trait Bar {
    type Output;
}

impl<T> Foo for dyn Bar<Output = T>
where
    Self: Sized,
{
    fn do_stuff() -> Self {
        todo!()
    }
}

fn main() {}
