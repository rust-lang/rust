use crate::*;

use crate::shims::unix::fs::FileDescriptor;

use rustc_data_structures::fx::FxHashMap;
use std::io;

/// An `Epoll` file descriptor connects file handles and epoll events
#[derive(Clone, Debug, Default)]
pub struct Epoll {
    /// The file descriptors we are watching, and what we are watching for.
    pub file_descriptors: FxHashMap<i32, EpollEvent>,
}

/// Epoll Events associate events with data.
/// These fields are currently unused by miri.
/// This matches the `epoll_event` struct defined
/// by the epoll_ctl man page. For more information
/// see the man page:
///
/// <https://man7.org/linux/man-pages/man2/epoll_ctl.2.html>
#[derive(Clone, Debug)]
pub struct EpollEvent {
    pub events: u32,
    /// `Scalar<Provenance>` is used to represent the
    /// `epoll_data` type union.
    pub data: Scalar<Provenance>,
}

impl FileDescriptor for Epoll {
    fn name(&self) -> &'static str {
        "epoll"
    }

    fn dup(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(self.clone()))
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        Ok(Ok(0))
    }
}
