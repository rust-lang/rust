pub mod foreign_items;

mod env;
mod fd;
mod fs;
mod mem;
mod sync;
mod thread;
mod unnamed_socket;

mod android;
mod freebsd;
mod linux;
mod macos;
mod solarish;

// All the Unix-specific extension traits
pub use self::env::{EvalContextExt as _, UnixEnvVars};
pub use self::fd::{EvalContextExt as _, FdTable, FileDescription};
pub use self::fs::{DirTable, EvalContextExt as _};
pub use self::linux::epoll::EpollInterestTable;
pub use self::mem::EvalContextExt as _;
pub use self::sync::EvalContextExt as _;
pub use self::thread::EvalContextExt as _;
pub use self::unnamed_socket::EvalContextExt as _;

// Make up some constants.
const UID: u32 = 1000;
