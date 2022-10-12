#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![feature(associated_type_bounds)]
#![feature(box_patterns)]
#![feature(with_negative_coherence)]
#![recursion_limit = "256"]

mod helpers;
pub mod pp;
pub mod pprust;
