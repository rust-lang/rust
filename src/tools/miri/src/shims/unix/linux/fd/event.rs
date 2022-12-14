use crate::shims::unix::fs::FileDescriptor;

use rustc_const_eval::interpret::InterpResult;

use std::io;

/// A kind of file descriptor created by `eventfd`.
/// The `Event` type isn't currently written to by `eventfd`.
/// The interface is meant to keep track of objects associated
/// with a file descriptor. For more information see the man
/// page below:
///
/// <https://man.netbsd.org/eventfd.2>
#[derive(Debug)]
pub struct Event {
    pub val: u32,
}

impl FileDescriptor for Event {
    fn name(&self) -> &'static str {
        "event"
    }

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(Event { val: self.val }))
    }

    fn is_tty(&self) -> bool {
        false
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        Ok(Ok(0))
    }
}
