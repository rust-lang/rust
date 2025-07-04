#![warn(clippy::arithmetic_side_effects)]

mod aarch64;
mod alloc;
mod backtrace;
mod files;
#[cfg(unix)]
mod native_lib;
mod unix;
mod wasi;
mod windows;
mod x86;

pub mod env;
pub mod extern_static;
pub mod foreign_items;
pub mod io_error;
pub mod os_str;
pub mod panic;
pub mod time;
pub mod tls;

pub use self::files::FdTable;
//#[cfg(target_os = "linux")]
//pub use self::native_lib::trace::{init_sv, register_retcode_sv};
pub use self::unix::{DirTable, EpollInterestTable};

/// What needs to be done after emulating an item (a shim or an intrinsic) is done.
pub enum EmulateItemResult {
    /// The caller is expected to jump to the return block.
    NeedsReturn,
    /// The caller is expected to jump to the unwind block.
    NeedsUnwind,
    /// Jumping to the next block has already been taken care of.
    AlreadyJumped,
    /// The item is not supported.
    NotSupported,
}
