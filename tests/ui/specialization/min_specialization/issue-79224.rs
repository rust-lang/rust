#![feature(min_specialization)]
use std::fmt::{self, Display};

pub enum Cow<'a, B: ?Sized + 'a, O = <B as ToOwned>::Owned>
where
    B: ToOwned,
{
    Borrowed(&'a B),
    Owned(O),
}

impl ToString for Cow<'_, str> {
    fn to_string(&self) -> String {
        String::new()
    }
}

impl<B: ?Sized> Display for Cow<'_, B> {
    //~^ ERROR trait `Clone` is not implemented for `B`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //~^ ERROR trait `Clone` is not implemented for `B`
        //~| ERROR trait `Clone` is not implemented for `B`
        //~| ERROR trait `Clone` is not implemented for `B`
        write!(f, "foo")
    }
}

fn main() {}
