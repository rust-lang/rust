//! System bindings for the wasm/web platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for wasm. Note that this wasm is *not* the emscripten
//! wasm, so we have no runtime here.
//!
//! This is all super highly experimental and not actually intended for
//! wide/production use yet, it's still all in the experimental category. This
//! will likely change over time.
//!
//! Currently all functions here are basically stubs that immediately return
//! errors. The hope is that with a portability lint we can turn actually just
//! remove all this and just omit parts of the standard library if we're
//! compiling for wasm. That way it's a compile time error for something that's
//! guaranteed to be a runtime error!

#[path = "../unix/alloc.rs"]
pub mod alloc;
pub mod args;
pub mod env;
pub mod fd;
pub mod fs;
#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;
pub mod io;

pub mod net;
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
pub mod thread;
#[path = "../unsupported/thread_local_dtor.rs"]
pub mod thread_local_dtor;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
pub mod time;

cfg_if::cfg_if! {
    if #[cfg(not(target_feature = "atomics"))] {
        #[path = "../unsupported/thread_parking.rs"]
        pub mod thread_parking;
    }
}

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;
pub use common::*;

mod helpers;
// These exports are listed individually to work around Rust's glob import
// conflict rules. If we glob export `helpers` and `common` together, then
// the compiler complains about conflicts.
pub use helpers::abort_internal;
pub use helpers::decode_error_kind;
use helpers::err2io;
pub use helpers::hashmap_random_keys;
pub use helpers::is_interrupted;
