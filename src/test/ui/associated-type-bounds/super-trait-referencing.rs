// check-pass

// The goal of this test is to ensure that T: Bar<T::Item>
// in the where clause does not cycle

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
