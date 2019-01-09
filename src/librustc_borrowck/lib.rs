#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![allow(non_camel_case_types)]

#![feature(nll)]
#![feature(quote)]

#![recursion_limit="256"]

#[macro_use] extern crate log;
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;
extern crate rustc_data_structures;

// for "clarity", rename the graphviz crate to dot; graphviz within `borrowck`
// refers to the borrowck-specific graphviz adapter traits.
extern crate graphviz as dot;
#[macro_use]
extern crate rustc;
extern crate rustc_mir;

pub use borrowck::check_crate;
pub use borrowck::build_borrowck_dataflow_data_for_fn;

mod borrowck;

pub mod graphviz;

mod dataflow;

pub use borrowck::provide;
