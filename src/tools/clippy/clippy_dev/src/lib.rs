#![feature(let_chains)]
#![feature(rustc_private)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications
)]
#![allow(clippy::missing_panics_doc)]

// The `rustc_driver` crate seems to be required in order to use the `rust_lexer` crate.
#[allow(unused_extern_crates)]
extern crate rustc_driver;
extern crate rustc_lexer;
extern crate rustc_literal_escaper;

pub mod dogfood;
pub mod fmt;
pub mod lint;
pub mod new_lint;
pub mod release;
pub mod serve;
pub mod setup;
pub mod sync;
pub mod update_lints;
pub mod utils;
