pub mod foreign_items;

mod env;
mod fd;
mod fs;
mod mem;
mod socket;
mod sync;
mod thread;

mod android;
mod freebsd;
mod linux;
mod macos;
mod solarish;

pub use env::UnixEnvVars;
pub use fd::{FdTable, FileDescription};
pub use fs::DirTable;
// All the Unix-specific extension traits
pub use env::EvalContextExt as _;
pub use fd::EvalContextExt as _;
pub use fs::EvalContextExt as _;
pub use mem::EvalContextExt as _;
pub use socket::EvalContextExt as _;
pub use sync::EvalContextExt as _;
pub use thread::EvalContextExt as _;

// Make up some constants.
const UID: u32 = 1000;
