pub mod foreign_items;

mod fd;
mod fs;
mod mem;
mod socket;
mod sync;
mod thread;

mod freebsd;
mod linux;
mod macos;

pub use fd::{FdTable, FileDescriptor};
pub use fs::DirTable;
// All the unix-specific extension traits
pub use fd::EvalContextExt as _;
pub use fs::EvalContextExt as _;
pub use mem::EvalContextExt as _;
pub use socket::EvalContextExt as _;
pub use sync::EvalContextExt as _;
pub use thread::EvalContextExt as _;

// Make up some constants.
const UID: u32 = 1000;
