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

pub use self::env::UnixEnvVars;
pub use self::fd::{FdTable, FileDescription};
pub use self::fs::DirTable;
pub use self::linux::epoll::EpollInterestTable;
// All the Unix-specific extension traits
pub use self::env::EvalContextExt as _;
pub use self::fd::EvalContextExt as _;
pub use self::fs::EvalContextExt as _;
pub use self::mem::EvalContextExt as _;
pub use self::sync::EvalContextExt as _;
pub use self::thread::EvalContextExt as _;
pub use self::unnamed_socket::EvalContextExt as _;

// Make up some constants.
const UID: u32 = 1000;
