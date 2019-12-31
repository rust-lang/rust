#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

pub use crate::sys_common::thread_local::register_dtor_fallback as register_dtor;
