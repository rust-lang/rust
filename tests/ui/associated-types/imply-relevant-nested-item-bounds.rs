//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

trait Foo
where
    Self::Iterator: Iterator,
    <Self::Iterator as Iterator>::Item: Bar,
{
    type Iterator;

    fn iter() -> Self::Iterator;
}

trait Bar {
    fn bar(&self);
}

fn x<T: Foo>() {
    T::iter().next().unwrap().bar();
}

fn main() {}
