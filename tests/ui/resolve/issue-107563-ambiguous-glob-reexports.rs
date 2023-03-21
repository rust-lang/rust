#![deny(ambiguous_glob_reexports)]

pub mod foo {
    pub type X = u8;
}

pub mod bar {
    pub type X = u8;
    pub type Y = u8;
}

pub use foo::*;
//~^ ERROR ambiguous glob re-exports
pub use bar::*;

mod ambiguous {
    mod m1 { pub type A = u8; }
    mod m2 { pub type A = u8; }
    pub use self::m1::*;
    //~^ ERROR ambiguous glob re-exports
    pub use self::m2::*;
}

pub mod single {
    pub use ambiguous::A;
    //~^ ERROR `A` is ambiguous
}

pub mod glob {
    pub use ambiguous::*;
}

pub fn main() {}
