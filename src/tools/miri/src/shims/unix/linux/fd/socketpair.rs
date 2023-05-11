use crate::*;

use crate::shims::unix::fs::FileDescriptor;

use std::io;

/// Pair of connected sockets.
///
/// We currently don't allow sending any data through this pair, so this can be just a dummy.
#[derive(Debug)]
pub struct SocketPair;

impl FileDescriptor for SocketPair {
    fn name(&self) -> &'static str {
        "socketpair"
    }

    fn dup(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(SocketPair))
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        Ok(Ok(0))
    }
}
