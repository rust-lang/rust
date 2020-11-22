#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(core_intrinsics)]
#![feature(crate_visibility_modifier)]
#![feature(drain_filter)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(once_cell)]
#![feature(or_patterns)]
#![feature(proc_macro_internals)]
#![feature(min_specialization)]
#![feature(stmt_expr_attributes)]
#![feature(try_blocks)]
#![feature(never_type)]
#![recursion_limit = "256"]

extern crate proc_macro;

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_data_structures;

pub use rmeta::{provide, provide_extern};

mod dependency_format;
mod foreign_modules;
mod link_args;
mod native_libs;
mod rmeta;

pub mod creader;
pub mod dynamic_lib;
pub mod locator;
