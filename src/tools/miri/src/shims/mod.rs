#![warn(clippy::arithmetic_side_effects)]

mod alloc;
mod backtrace;
#[cfg(target_os = "linux")]
mod native_lib;
mod unix;
mod wasi;
mod windows;
mod x86;

pub mod env;
pub mod extern_static;
pub mod foreign_items;
pub mod os_str;
pub mod panic;
pub mod time;
pub mod tls;

pub use unix::{DirTable, FdTable};

/// What needs to be done after emulating an item (a shim or an intrinsic) is done.
pub enum EmulateItemResult {
    /// The caller is expected to jump to the return block.
    NeedsJumping,
    /// Jumping has already been taken care of.
    AlreadyJumped,
    /// The item is not supported.
    NotSupported,
}
