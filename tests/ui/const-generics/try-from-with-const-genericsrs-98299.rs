// https://github.com/rust-lang/rust/issues/98299
use std::convert::TryFrom;

pub fn test_usage(p: ()) {
    SmallCString::try_from(p).map(|cstr| cstr);
    //~^ ERROR: type annotations needed
    //~| ERROR: type annotations needed
    //~| ERROR: type annotations needed
}

pub struct SmallCString<const N: usize> {}

impl<const N: usize> TryFrom<()> for SmallCString<N> {
    type Error = ();

    fn try_from(path: ()) -> Result<Self, Self::Error> {
        unimplemented!();
    }
}

fn main() {}
