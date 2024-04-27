/* This crate declares an item that is unnamed.
 * Its only public path is through `prelude::*`. */

pub struct S;

mod m {
    pub trait Tr { fn method(&self); }
    impl Tr for crate::S { fn method(&self) {} }
}

pub mod prelude {
    pub use crate::m::Tr as _;
}
