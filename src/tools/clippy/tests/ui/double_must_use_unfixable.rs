#![warn(clippy::double_must_use)]
#![expect(clippy::result_unit_err)]
#![feature(never_type)]

#[cfg_attr(all(), must_use, deprecated)]
pub fn issue_12320() -> Result<(), ()> {
    //~^ double_must_use
    unimplemented!();
}

#[cfg_attr(all(), deprecated, must_use)]
pub fn issue_12320_2() -> Result<(), ()> {
    //~^ double_must_use
    unimplemented!();
}

fn main() {}
