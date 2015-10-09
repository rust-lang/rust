pub use imp::rt as imp;

pub mod traits {
    pub use super::Runtime as sys_Runtime;
}

pub mod prelude {
    pub use super::imp::Runtime;
    pub use super::traits::*;
}

use core::any::Any;
use core::fmt;

pub trait Runtime {
    unsafe fn run_main<R, F: FnOnce() -> R>(f: F, argc: isize, argv: *const *const u8) -> R;
    unsafe fn run_thread<R, F: FnOnce() -> R>(f: F) -> R;
    unsafe fn thread_cleanup();
    unsafe fn cleanup();

    fn on_panic(msg: &(Any + Send), file: &'static str, line: u32);
    fn min_stack() -> usize;
    fn abort(args: fmt::Arguments) -> !;
}
