#![deny(rustdoc::unused_reexport_documentation)]

pub mod a {
    pub struct Foo;
}

mod b {
    pub struct Priv;
}

pub mod c {
    /// Error!
    //~^ ERROR
    pub use a::Foo as Bar;
    /// No error for this one.
    #[doc(inline)]
    pub use a::Foo as Bar2;
    /// No error for this one.
    pub use b::Priv as Priv;
}

/// No error for this one.
pub use std::option::Option as AnotherOption;
/// Error!
//~^ ERROR
pub use std::option::*;
/// Error!
//~^ ERROR
pub use b::*;
