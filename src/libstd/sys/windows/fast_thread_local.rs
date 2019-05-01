#![unstable(feature = "thread_local_internals", issue = "0")]
#![cfg(target_thread_local)]

pub use crate::sys_common::thread_local::register_dtor_fallback as register_dtor;
