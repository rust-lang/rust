// check-pass
trait Foo {
    type Item;
}

trait Bar<T> {}

fn baz<T>()
where
    T: Foo,
    T: Bar<T::Item>,
{
}

fn main() {}
