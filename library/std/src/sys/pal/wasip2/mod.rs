//! System bindings for the wasi preview 2 target.
//!
//! This is the next evolution of the original wasi target, and is intended to
//! replace that target over time.
//!
//! To begin with, this target mirrors the wasi target 1 to 1, but over
//! time this will change significantly.

#[path = "../unix/alloc.rs"]
pub mod alloc;
#[path = "../wasi/args.rs"]
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
#[path = "../wasi/env.rs"]
pub mod env;
#[path = "../wasi/fd.rs"]
pub mod fd;
#[path = "../wasi/fs.rs"]
pub mod fs;
#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;
#[path = "../wasi/io.rs"]
pub mod io;

#[path = "../wasi/net.rs"]
pub mod net;
#[path = "../wasi/os.rs"]
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
#[path = "../wasi/stdio.rs"]
pub mod stdio;
#[path = "../wasi/thread.rs"]
pub mod thread;
#[path = "../unsupported/thread_local_dtor.rs"]
pub mod thread_local_dtor;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
#[path = "../wasi/time.rs"]
pub mod time;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "atomics")] {
        compile_error!("The wasm32-wasip2 target does not support atomics");
    } else {
        #[path = "../unsupported/locks/mod.rs"]
        pub mod locks;
        #[path = "../unsupported/once.rs"]
        pub mod once;
        #[path = "../unsupported/thread_parking.rs"]
        pub mod thread_parking;
    }
}

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;
pub use common::*;

#[path = "../wasi/helpers.rs"]
mod helpers;
// These exports are listed individually to work around Rust's glob import
// conflict rules. If we glob export `helpers` and `common` together, then
// the compiler complains about conflicts.
pub use helpers::abort_internal;
pub use helpers::decode_error_kind;
use helpers::err2io;
pub use helpers::hashmap_random_keys;
pub use helpers::is_interrupted;
