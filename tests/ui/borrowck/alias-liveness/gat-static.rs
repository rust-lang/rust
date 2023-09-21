// check-pass

trait Foo {
    type Assoc<'a>
    where
        Self: 'a;

    fn assoc(&mut self) -> Self::Assoc<'_>;
}

fn test<T>(mut t: T)
where
    T: Foo,
    for<'a> T::Assoc<'a>: 'static,
{
    let a = t.assoc();
    let b = t.assoc();
}

fn main() {}
