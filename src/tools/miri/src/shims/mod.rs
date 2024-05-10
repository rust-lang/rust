#![warn(clippy::arithmetic_side_effects)]

mod alloc;
mod backtrace;
pub mod foreign_items;
#[cfg(target_os = "linux")]
pub mod native_lib;
pub mod unix;
pub mod windows;
mod x86;

pub mod env;
pub mod extern_static;
pub mod os_str;
pub mod panic;
pub mod time;
pub mod tls;

/// What needs to be done after emulating an item (a shim or an intrinsic) is done.
pub enum EmulateItemResult {
    /// The caller is expected to jump to the return block.
    NeedsJumping,
    /// Jumping has already been taken care of.
    AlreadyJumped,
    /// The item is not supported.
    NotSupported,
}
