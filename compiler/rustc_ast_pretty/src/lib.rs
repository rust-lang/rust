#![allow(internal_features)]
#![feature(rustdoc_internals)]
#![doc(rust_logo)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![feature(box_patterns)]
#![recursion_limit = "256"]

mod helpers;
pub mod pp;
pub mod pprust;
