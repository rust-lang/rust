#![feature(never_type)]
#![warn(clippy::infallible_try_from)]

use std::convert::Infallible;

struct MyStruct(i32);

impl TryFrom<i8> for MyStruct {
    //~^ infallible_try_from
    type Error = !;
    fn try_from(other: i8) -> Result<Self, !> {
        Ok(Self(other.into()))
    }
}

impl TryFrom<i16> for MyStruct {
    //~^ infallible_try_from
    type Error = Infallible;
    fn try_from(other: i16) -> Result<Self, Infallible> {
        Ok(Self(other.into()))
    }
}

impl TryFrom<i64> for MyStruct {
    type Error = i64;
    fn try_from(other: i64) -> Result<Self, i64> {
        Ok(Self(i32::try_from(other).map_err(|_| other)?))
    }
}

fn main() {
    // test code goes here
}
