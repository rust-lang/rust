#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![feature(decl_macro)]
#![feature(extract_if)]
#![feature(coroutines)]
#![feature(iter_from_coroutine)]
#![feature(let_chains)]
#![feature(if_let_guard)]
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
extern crate tracing;

pub use rmeta::provide;
use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;

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
pub use rmeta::{encode_metadata, rendered_const, EncodedMetadata, METADATA_HEADER};

fluent_messages! { "../messages.ftl" }
