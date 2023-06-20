pub mod dlsym;
pub mod foreign_items;

mod fs;
mod mem;
mod sync;
mod thread;

mod android;
mod freebsd;
mod linux;
mod macos;

pub use fs::{DirHandler, FileHandler};

// Make up some constants.
const UID: u32 = 1000;
