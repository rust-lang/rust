use crate::{
    io,
    pipe::{PipeReader, PipeWriter},
    process::Stdio,
};

pub(crate) use crate::sys::pipe::AnonPipe;

#[inline]
pub(crate) fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    Err(io::Error::UNSUPPORTED_PLATFORM)
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeReader> for Stdio {
    fn from(pipe: PipeReader) -> Self {
        pipe.0.diverge()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl From<PipeWriter> for Stdio {
    fn from(pipe: PipeWriter) -> Self {
        pipe.0.diverge()
    }
}
