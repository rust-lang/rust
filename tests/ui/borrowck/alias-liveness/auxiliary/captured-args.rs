//@ edition: 2024

pub fn late<'a>(_: &'a String) -> impl Sized + use<'a> {}

pub fn early<'a: 'a>(_: &'a String) -> impl Sized + use<'a> {}

pub fn type_param<T>(_: T) -> impl Sized {}

pub fn uncaptured(_: &mut String) -> impl Sized + use<> {}

pub trait Captures {
    fn late<'a>(_: &'a String) -> impl Sized;
    fn early<'a: 'a>(_: &'a String) -> impl Sized;
    fn uncaptured(_: &mut String) -> impl Sized + use<Self>;
}
