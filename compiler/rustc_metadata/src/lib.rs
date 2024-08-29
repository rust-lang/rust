// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::potential_query_instability)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(coroutines)]
#![feature(decl_macro)]
#![feature(error_iter)]
#![feature(extract_if)]
#![feature(if_let_guard)]
#![feature(iter_from_coroutine)]
#![feature(let_chains)]
#![feature(macro_metavar_expr)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(proc_macro_internals)]
#![feature(rustdoc_internals)]
#![feature(trusted_len)]
// tidy-alphabetical-end

extern crate proc_macro;

pub use rmeta::provide;

mod dependency_format;
mod foreign_modules;
mod native_libs;
mod rmeta;

pub mod creader;
pub mod errors;
pub mod fs;
pub mod locator;

pub use creader::{load_symbol_from_dylib, DylibError};
pub use fs::{emit_wrapper_file, METADATA_FILENAME};
pub use native_libs::find_native_static_library;
pub use rmeta::{encode_metadata, rendered_const, EncodedMetadata, METADATA_HEADER};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
