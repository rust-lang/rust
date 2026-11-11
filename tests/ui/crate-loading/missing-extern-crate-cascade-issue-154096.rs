//@ edition: 2024

extern crate missing_154096;
//~^ ERROR can't find crate for `missing_154096`

mod defs {
    pub use missing_154096::Thing;
}

pub use defs::Thing;

mod first {
    use crate::Thing;

    pub fn take(_: Thing) {}
}

mod second {
    use crate::Thing;

    pub fn take(_: Thing) {}
}

fn main() {}
