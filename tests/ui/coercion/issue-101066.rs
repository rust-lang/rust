//@ check-pass

use std::convert::TryFrom;

pub trait FieldElement {
    type Integer: TryFrom<usize, Error = std::num::TryFromIntError>;

    fn valid_integer_try_from<N>(i: N) -> Result<Self::Integer, ()>
    where
        Self::Integer: TryFrom<N>,
    {
        Self::Integer::try_from(i).map_err(|_| ())
    }
}

fn main() {}
