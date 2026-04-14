use abi::types::StreamId;
use abi::errors::Errno;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamHandle {
    pub id: StreamId,
}

/// Open a named stream.
///
/// Returns a StreamHandle on success.
/// Stub: returns Errno::NotSupported currently.
pub fn open(_name: &str) -> Result<StreamHandle, Errno> {
    // Stub implementation
    Err(Errno::NotSupported)
}

impl StreamHandle {
    /// Read from the stream into the provided buffer.
    pub fn read(&mut self, _buf: &mut [u8]) -> Result<usize, Errno> {
        // Stub implementation
        Err(Errno::NotSupported)
    }
}

/// Poll multiple streams.
pub fn poll(_handles: &mut [StreamHandle], _timeout_ms: u32) -> Result<usize, Errno> {
    // Stub implementation
    Err(Errno::NotSupported)
}
