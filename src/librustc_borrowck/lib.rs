#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![allow(non_camel_case_types)]
#![deny(rust_2018_idioms)]

#![feature(nll)]

#![recursion_limit="256"]

#[macro_use]
extern crate rustc;

pub use borrowck::check_crate;
pub use borrowck::build_borrowck_dataflow_data_for_fn;

mod borrowck;

pub mod graphviz;

mod dataflow;

pub use borrowck::provide;
