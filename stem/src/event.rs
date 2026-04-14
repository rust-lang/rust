use abi::types::{WatchId, EventHeader};
use abi::errors::Errno;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatchHandle {
    pub id: WatchId,
}

/// Subscribe to a watch selector.
/// Selector is opaque bytes for now.
pub fn subscribe(_selector: &[u8]) -> Result<WatchHandle, Errno> {
    Err(Errno::NotSupported)
}

impl WatchHandle {
    /// Read events from the watch handle.
    pub fn read_events(&mut self, _buf: &mut [u8]) -> Result<usize, Errno> {
        Err(Errno::NotSupported)
    }
}

/// Helper to parse an event header from a buffer.
/// Returns (Header, RemainingBytes) if successful.
pub fn parse_header(buf: &[u8]) -> Option<(EventHeader, &[u8])> {
    if buf.len() < core::mem::size_of::<EventHeader>() {
        return None;
    }
    // Safety: EventHeader is POT and we checked length.
    // Note: This naive cast assumes alignment, which might be unsafe depending on buffer source.
    // For now, in a stub, we'll just use read_unaligned or similar if we could.
    // But since this is no_std and we want to be safe, let's just copy bytes.
    let mut header_bytes = [0u8; core::mem::size_of::<EventHeader>()];
    header_bytes.copy_from_slice(&buf[..core::mem::size_of::<EventHeader>()]);
    let header = unsafe { core::mem::transmute::<[u8; core::mem::size_of::<EventHeader>()], EventHeader>(header_bytes) };
    
    Some((header, &buf[core::mem::size_of::<EventHeader>()..]))
}
