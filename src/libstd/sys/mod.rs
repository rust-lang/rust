#![allow(missing_docs)]
#![doc(hidden)]

#[macro_use]
pub mod inner;
pub mod os;

#[cfg(target_family = "windows")]
mod wtf8;

#[macro_use]
pub mod common;

#[cfg(target_family = "unix")] pub mod unix;
#[cfg(target_family = "windows")] pub mod windows;

#[cfg(target_family = "unix")] pub use self::unix as target;
#[cfg(target_family = "windows")] pub use self::windows as target;

pub use self::target::{
    time,
    unwind,
    sync,
    backtrace,
    path,
    thread,
    rand,
    dynamic_lib,
    fs,
    stdio,
    env,
    rt,
    process,
    os_str,
    error,
    thread_local,
    net,
};
