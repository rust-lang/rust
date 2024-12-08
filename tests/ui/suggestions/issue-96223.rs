// Previously ICEd because we didn't properly track binders in suggestions
//@ check-fail

pub trait Foo<'de>: Sized {}

pub trait Bar<'a>: 'static {
    type Inner: 'a;
}

pub trait Fubar {
    type Bar: for<'a> Bar<'a>;
}

pub struct Baz<T>(pub T);

impl<'de, T> Foo<'de> for Baz<T> where T: Foo<'de> {}

struct Empty;

impl<M> Dummy<M> for Empty
where
    M: Fubar,
    for<'de> Baz<<M::Bar as Bar<'de>>::Inner>: Foo<'de>,
{
}

pub trait Dummy<M>
where
    M: Fubar,
{
}

pub struct EmptyBis<'a>(&'a [u8]);

impl<'a> Bar<'a> for EmptyBis<'static> {
    type Inner = EmptyBis<'a>;
}

pub struct EmptyMarker;

impl Fubar for EmptyMarker {
    type Bar = EmptyBis<'static>;
}

fn icey_bounds<D: Dummy<EmptyMarker>>(p: &D) {}

fn trigger_ice() {
    let p = Empty;
    icey_bounds(&p); //~ERROR the trait bound
}

fn main() {}
