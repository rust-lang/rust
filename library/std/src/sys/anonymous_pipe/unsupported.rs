use crate::io::{self, PipeReader, PipeWriter};
use crate::process::Stdio;
pub use crate::sys::pipe::AnonPipe;

#[inline]
pub fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    Err(io::Error::UNSUPPORTED_PLATFORM)
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeReader> for Stdio {
    fn from(pipe: PipeReader) -> Self {
        pipe.0.diverge()
    }
}

#[stable(feature = "anonymous_pipe", since = "CURRENT_RUSTC_VERSION")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        pipe.0.diverge()
    }
}
