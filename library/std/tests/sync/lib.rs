#![feature(lazy_get)]
#![feature(mapped_lock_guards)]
#![feature(mpmc_channel)]
#![feature(once_cell_try)]
#![feature(lock_value_accessors)]
#![feature(reentrant_lock)]
#![feature(rwlock_downgrade)]
#![feature(std_internals)]
#![allow(internal_features)]

mod barrier;
mod condvar;
mod lazy_lock;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod mpmc;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod mpsc;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod mpsc_sync;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod mutex;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod once;
mod once_lock;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod reentrant_lock;
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod rwlock;

#[path = "../common/mod.rs"]
mod common;
