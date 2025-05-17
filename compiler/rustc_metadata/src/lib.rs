// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(coroutines)]
#![feature(decl_macro)]
#![feature(error_iter)]
#![feature(file_buffered)]
#![feature(if_let_guard)]
#![feature(iter_from_coroutine)]
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

pub use creader::{DylibError, load_symbol_from_dylib};
pub use fs::{METADATA_FILENAME, emit_wrapper_file};
pub use native_libs::{
    NativeLibSearchFallback, find_native_static_library, try_find_native_dynamic_library,
    try_find_native_static_library, walk_native_lib_search_dirs,
};
pub use rmeta::{EncodedMetadata, METADATA_HEADER, encode_metadata, rendered_const};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
