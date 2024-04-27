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
    //~^ ERROR: the trait bound `B: Clone` is not satisfied [E0277]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //~^ ERROR: the trait bound `B: Clone` is not satisfied [E0277]
        //~| ERROR: the trait bound `B: Clone` is not satisfied [E0277]
        //~| ERROR: the trait bound `B: Clone` is not satisfied [E0277]
        write!(f, "foo")
    }
}

fn main() {}
