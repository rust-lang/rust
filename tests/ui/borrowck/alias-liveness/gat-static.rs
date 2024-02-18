//@ check-pass

trait Foo {
    type Assoc<'a>
    where
        Self: 'a;

    fn assoc(&mut self) -> Self::Assoc<'_>;
}

fn overlapping_mut<T>(mut t: T)
where
    T: Foo,
    for<'a> T::Assoc<'a>: 'static,
{
    let a = t.assoc();
    let b = t.assoc();
}

fn live_past_borrow<T>(mut t: T)
where
    T: Foo,
    for<'a> T::Assoc<'a>: 'static {
    let x = t.assoc();
    drop(t);
    drop(x);
}

fn main() {}
