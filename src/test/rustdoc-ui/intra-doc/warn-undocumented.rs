// aux-build:dep1.rs

#![deny(rustdoc::private_intra_doc_links)]
//~^ NOTE defined here

//! Link to [dep1]
//~^ ERROR will not have documentation generated
//~| NOTE will not be documented
//~| crate `dep1`, which will not be documented

//! Link to [S]
//~^ ERROR will not have documentation generated
//~| NOTE will not be documented
//~| NOTE may be in a private module

//! Link to [T]
//~^ ERROR will not have documentation generated
//~| NOTE will not be documented

//! Link to [secret::U]
//~^ ERROR will not have documentation generated
//~| NOTE will not be documented

extern crate dep1;

#[doc(no_inline)]
pub use inner::S;

mod inner {
    pub struct S;
}

#[doc(hidden)]
//~^ NOTE doc(hidden)
pub struct T;

#[doc(hidden)]
//~^ NOTE `secret` is hidden
pub mod secret {
//~^ NOTE doc(hidden)
    pub struct U;
}