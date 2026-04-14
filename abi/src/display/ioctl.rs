//! Display Device Ops (IOCTLs)

use super::types::{BufferHandle, BufferId, CommitRequest, DisplayInfo};

/// Retrieve display device information and capabilities.
/// Output: DisplayInfo
pub const DISPLAY_OP_GET_INFO: u32 = 1;

/// Import a shared memory buffer.
/// Input: BufferHandle
/// Output: BufferId (via u32 return value or out_ptr)
pub const DISPLAY_OP_IMPORT_BUFFER: u32 = 2;

/// Release an imported buffer.
/// Input: BufferId
pub const DISPLAY_OP_RELEASE_BUFFER: u32 = 3;

/// Atomic presentation commit.
/// Input: CommitRequest
pub const DISPLAY_OP_COMMIT: u32 = 4;

/// Set display mode.
/// Input: DisplayMode
pub const DISPLAY_OP_SET_MODE: u32 = 5;
