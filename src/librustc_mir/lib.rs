#![feature(
    i128_type,
    rustc_private,
    conservative_impl_trait,
)]

// From rustc.
#[macro_use]
extern crate log;
extern crate log_settings;
#[macro_use]
extern crate rustc;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate syntax;

// From crates.io.
extern crate byteorder;
#[macro_use]
extern crate lazy_static;
extern crate regex;

pub mod interpret;
