#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(decl_macro)]
#![feature(drain_filter)]
#![feature(generators)]
#![feature(iter_from_generator)]
#![feature(let_chains)]
#![feature(proc_macro_internals)]
#![feature(macro_metavar_expr)]
#![feature(min_specialization)]
#![feature(slice_as_chunks)]
#![feature(trusted_len)]
#![feature(try_blocks)]
#![feature(never_type)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

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
use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_macros::fluent_messages;

mod dependency_format;
mod foreign_modules;
mod native_libs;
mod rmeta;

pub mod creader;
pub mod errors;
pub mod fs;
pub mod locator;

pub use fs::{emit_wrapper_file, METADATA_FILENAME};
pub use native_libs::find_native_static_library;
pub use rmeta::{encode_metadata, EncodedMetadata, METADATA_HEADER};

fluent_messages! { "../messages.ftl" }
