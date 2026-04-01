// https://github.com/rust-lang/rust/pull/113099#issuecomment-1637022296

mod a {
    pub struct Class(u16);
}

use a::Class;

mod gpos {
    use super::gsubgpos::*;
    use super::*;
    struct MarkRecord(Class);
    //~^ ERROR`Class` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

mod gsubgpos {
    use super::*;
    #[derive(Clone)]
    pub struct Class;
}

fn main() { }
