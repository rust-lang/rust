//! Display Event Types

/// Events delivered via the /dev/display/cardN/events node.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayEvent {
    /// Vertical Blanking interval started.
    VBlank { sequence: u64, timestamp_ns: u64 },
    /// A commit has completed and the buffer is now visible.
    PageFlipComplete {
        buffer_id: super::types::BufferId,
        timestamp_ns: u64,
    },
    /// A new output has been connected or removed.
    Hotplug,
}
