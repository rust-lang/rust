#![feature(rustc_private)]
#![warn(
    rust_2018_idioms,
    trivial_casts,
    trivial_numeric_casts,
    unused_lifetimes,
    unused_qualifications
)]
#![expect(clippy::must_use_candidate)]

extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod conf;
mod metadata;
pub mod types;

pub use conf::{Conf, get_configuration_metadata, lookup_conf_file, sanitize_explanation};
pub use metadata::ClippyConfiguration;
