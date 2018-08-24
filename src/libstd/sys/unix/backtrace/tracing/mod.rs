pub use self::imp::*;

#[cfg(not(all(target_os = "ios", target_arch = "arm")))]
#[path = "gcc_s.rs"]
mod imp;
#[cfg(all(target_os = "ios", target_arch = "arm"))]
#[path = "backtrace_fn.rs"]
mod imp;
