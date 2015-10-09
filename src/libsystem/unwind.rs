pub use imp::unwind as imp;

pub mod traits {
    pub use super::Unwind as sys_Unwind;
}

pub mod prelude {
    pub use super::imp::Unwind;
    pub use super::traits::*;
}

use core::any::Any;
use alloc::boxed::Box;
use core::fmt;

pub trait Unwind {
    fn begin_unwind_fmt(msg: fmt::Arguments, file_line: &(&'static str, u32)) -> !;
    fn begin_unwind<M: Any + Send>(msg: M, file_line: &(&'static str, u32)) -> !;

    fn panic_inc() -> usize;
    fn is_panicking() -> bool;

    unsafe fn try<F: FnOnce()>(f: F) -> Result<(), Box<Any + Send>>;
}
