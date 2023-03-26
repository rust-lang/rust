use crate::shims::unix::fs::FileDescriptor;

use rustc_const_eval::interpret::InterpResult;

use std::cell::Cell;
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
    /// The object contains an unsigned 64-bit integer (uint64_t) counter that is maintained by the
    /// kernel. This counter is initialized with the value specified in the argument initval.
    pub val: Cell<u64>,
}

impl FileDescriptor for Event {
    fn name(&self) -> &'static str {
        "event"
    }

    fn dup(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(Event { val: self.val.clone() }))
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        Ok(Ok(0))
    }

    /// A write call adds the 8-byte integer value supplied in
    /// its buffer to the counter.  The maximum value that may be
    /// stored in the counter is the largest unsigned 64-bit value
    /// minus 1 (i.e., 0xfffffffffffffffe).  If the addition would
    /// cause the counter's value to exceed the maximum, then the
    /// write either blocks until a read is performed on the
    /// file descriptor, or fails with the error EAGAIN if the
    /// file descriptor has been made nonblocking.

    /// A write fails with the error EINVAL if the size of the
    /// supplied buffer is less than 8 bytes, or if an attempt is
    /// made to write the value 0xffffffffffffffff.
    ///
    /// FIXME: use endianness
    fn write<'tcx>(
        &self,
        _communicate_allowed: bool,
        bytes: &[u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        let v1 = self.val.get();
        // FIXME handle blocking when addition results in exceeding the max u64 value
        // or fail with EAGAIN if the file descriptor is nonblocking.
        let v2 = v1.checked_add(u64::from_be_bytes(bytes.try_into().unwrap())).unwrap();
        self.val.set(v2);
        assert_eq!(8, bytes.len());
        Ok(Ok(8))
    }
}
