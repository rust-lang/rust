pub mod c {
    pub struct S {}
    pub struct TS();
    pub struct US;
    pub enum E {
        V {},
        TV(),
        UV,
    }

    pub struct Item;
}

pub mod xm1 {
    pub use ::c::*;
    pub type S = ::c::Item;
}
pub mod xm2 {
    pub use ::c::*;
    pub const S: ::c::Item = ::c::Item;
}

pub mod xm3 {
    pub use ::c::*;
    pub type TS = ::c::Item;
}
pub mod xm4 {
    pub use ::c::*;
    pub const TS: ::c::Item = ::c::Item;
}

pub mod xm5 {
    pub use ::c::*;
    pub type US = ::c::Item;
}
pub mod xm6 {
    pub use ::c::*;
    pub const US: ::c::Item = ::c::Item;
}

pub mod xm7 {
    pub use ::c::E::*;
    pub type V = ::c::Item;
}
pub mod xm8 {
    pub use ::c::E::*;
    pub const V: ::c::Item = ::c::Item;
}

pub mod xm9 {
    pub use ::c::E::*;
    pub type TV = ::c::Item;
}
pub mod xmA {
    pub use ::c::E::*;
    pub const TV: ::c::Item = ::c::Item;
}

pub mod xmB {
    pub use ::c::E::*;
    pub type UV = ::c::Item;
}
pub mod xmC {
    pub use ::c::E::*;
    pub const UV: ::c::Item = ::c::Item;
}
