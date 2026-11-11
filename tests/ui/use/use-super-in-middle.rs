pub mod x {
    pub use crate::x::super::{self as crate1}; //~ ERROR `super` in paths can only be used in start position
    pub use crate::x::self::super::{self as crate2}; //~ ERROR `self` in paths can only be used in start position

    pub fn foo() {}
}

fn main() {
    x::crate1::x::foo();
    x::crate2::x::foo();

    crate::x::super::x::foo(); //~ ERROR `super` in paths can only be used in start position
    crate::x::self::super::x::foo(); //~ ERROR `self` in paths can only be used in start position
}
