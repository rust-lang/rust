//@compile-flags: -Z deduplicate-diagnostics=yes
//~^ ERROR overflow

pub trait Archive {
    type Archived;
}

impl<'a> Archive for <&'a [u8] as Archive>::Archived {
    //~^ ERROR overflow
    //~| ERROR `&'a [u8]: Archive` is not satisfied
    //~| ERROR `&'a [u8]: Archive` is not satisfied
    type Archived = ();
}

fn main() {}
