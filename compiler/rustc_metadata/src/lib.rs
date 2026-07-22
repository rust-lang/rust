// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(error_iter)]
#![feature(file_buffered)]
#![feature(gen_blocks)]
#![feature(macro_metavar_expr)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(option_into_flat_iter)]
#![feature(proc_macro_internals)]
#![feature(trusted_len)]
// tidy-alphabetical-end

pub use rmeta::provide;

mod dependency_format;
mod eii;
mod foreign_modules;
mod host_dylib;
mod native_libs;
mod rmeta;

pub mod creader;
pub mod diagnostics;
pub mod fs;
pub mod locator;

pub use fs::{METADATA_FILENAME, emit_wrapper_file};
pub use host_dylib::{DylibError, load_symbol_from_dylib};
pub use native_libs::{
    NativeLibSearchFallback, find_bundled_library, find_native_static_library,
    try_find_native_dynamic_library, try_find_native_static_library, walk_native_lib_search_dirs,
};
pub use rmeta::{EncodedMetadata, METADATA_HEADER, ProcMacroKind, encode_metadata, rendered_const};
