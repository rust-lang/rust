#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(decl_macro)]
#![feature(drain_filter)]
#![feature(generators)]
#![feature(generic_associated_types)]
#![feature(iter_from_generator)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(once_cell)]
#![feature(proc_macro_internals)]
#![feature(macro_metavar_expr)]
#![feature(min_specialization)]
#![feature(slice_as_chunks)]
#![feature(trusted_len)]
#![feature(try_blocks)]
#![feature(never_type)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]

extern crate proc_macro;

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_data_structures;

#[macro_use]
extern crate tracing;

pub use rmeta::{provide, provide_extern};

mod dependency_format;
mod foreign_modules;
mod native_libs;
mod rmeta;

pub mod creader;
pub mod fs;
pub mod locator;

pub use fs::{emit_metadata, METADATA_FILENAME};
pub use rmeta::{encode_metadata, EncodedMetadata, METADATA_HEADER};
