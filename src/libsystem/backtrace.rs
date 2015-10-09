pub use imp::backtrace as imp;

pub mod traits {
    pub use super::{Backtrace as sys_Backtrace, BacktraceOutput as sys_BacktraceOutput};
}

pub mod prelude {
    pub use super::imp::Backtrace;
    pub use super::BacktraceOutput;
    pub use super::traits::*;
}

use error::prelude::*;
use core::fmt;

pub trait BacktraceOutput: fmt::Write {
    fn output(&mut self, idx: isize, addr: *mut (), s: Option<&[u8]>) -> Result<()>;
    fn output_fileline(&mut self, file: &[u8], line: i32, more: bool) -> Result<()>;
}

pub trait Backtrace {
    // const fn new() -> Self;

    fn log_enabled() -> bool;

    fn write<O: BacktraceOutput>(&mut self, w: &mut O) -> Result<()>;
}
