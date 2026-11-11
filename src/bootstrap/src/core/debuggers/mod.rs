//! Code for discovering debuggers and debugger-related configuration, so that
//! it can be passed to compiletest when running debuginfo tests.

pub(crate) use self::cdb::{Cdb, discover_cdb};
pub(crate) use self::gdb::{Gdb, discover_gdb};
pub(crate) use self::lldb::{Lldb, discover_lldb};

mod cdb;
mod gdb;
mod lldb;
