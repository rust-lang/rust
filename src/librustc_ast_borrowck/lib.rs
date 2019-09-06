#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![allow(non_camel_case_types)]

#![feature(in_band_lifetimes)]
#![feature(nll)]

#![recursion_limit="256"]

#[macro_use]
extern crate rustc;

pub use borrowck::check_crate;
pub use borrowck::build_borrowck_dataflow_data_for_fn;

mod borrowck;

pub mod graphviz;

mod dataflow;
pub mod cfg;

pub use borrowck::provide;
