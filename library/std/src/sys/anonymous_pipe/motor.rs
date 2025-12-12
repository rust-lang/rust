use crate::io;
use crate::sys::IntoInner;
use crate::sys::fd::FileDesc;
use crate::sys::pipe::anon_pipe;

pub type AnonPipe = FileDesc;

#[inline]
pub fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    anon_pipe().map(|(rx, wx)| (rx.into_inner(), wx.into_inner()))
}
