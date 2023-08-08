//! Implements thread-local destructors that are not associated with any
//! particular data.

#![unstable(feature = "thread_local_internals", issue = "none")]
#![cfg(target_thread_local)]

pub use super::thread_local_key::register_keyless_dtor as register_dtor;
