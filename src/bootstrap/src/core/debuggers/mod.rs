//! Code for discovering debuggers and debugger-related configuration, so that
//! it can be passed to compiletest when running debuginfo tests.

pub(crate) use self::android::{Android, discover_android};
pub(crate) use self::gdb::{Gdb, discover_gdb};
pub(crate) use self::lldb::{Lldb, discover_lldb};

mod android;
mod gdb;
mod lldb;
