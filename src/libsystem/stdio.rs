pub use imp::stdio as imp;

pub mod traits {
    pub use super::Stdio as sys_Stdio;
}

pub mod prelude {
    pub use super::imp::{Stdio, Stdin, Stdout, Stderr};
    pub use super::traits::*;
    pub use super::{handle_ebadf, dumb_print};
}

use error::prelude::*;
use io::prelude::*;
use core::fmt;

pub trait Stdio {
    #[inline(always)] fn ebadf() -> i32;

    type Stdin: Read;
    type Stdout: Write;
    type Stderr: Write;

    fn stdin() -> Result<Self::Stdin> where Self::Stdin: Sized;
    fn stdout() -> Result<Self::Stdout> where Self::Stdout: Sized;
    fn stderr() -> Result<Self::Stderr> where Self::Stderr: Sized;
}

pub fn handle_ebadf<S: Stdio + ?Sized, T>(r: Result<T>, default: T) -> Result<T> {
    match r {
        Err(ref e) if e.code() == S::ebadf() => Ok(default),
        r => r
    }
}

pub fn dumb_print(args: fmt::Arguments) {
    use stdio::prelude::*;

    let _ = Stdio::stderr().map(|mut stderr| fmt::Write::write_fmt(&mut stderr.fmt_writer(), args));
}
