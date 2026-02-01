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
}

mod gsubgpos {
    use super::*;
    #[derive(Clone)]
    pub struct Class;
}

fn main() { }
