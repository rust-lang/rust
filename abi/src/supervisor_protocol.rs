//! Supervisor Protocol for Sovereign Driver Registration
//!
//! This protocol defines the handshake between system drivers and the supervisor (sprout).
//! It replaces the "land-rush" model where drivers mount themselves into /dev.
//!
//! # Message Overview
//!
//! | Constant            | Value  | Direction            | Meaning                                    |
//! |---------------------|--------|----------------------|--------------------------------------------|
//! | `MSG_BIND_READY`    | 0x8001 | Driver → Supervisor  | Driver is ready; sends VFS provider handle |
//! | `MSG_BIND_ASSIGNED` | 0x8002 | Supervisor → Driver  | Registration accepted; path assigned       |
//! | `MSG_BIND_FAILED`   | 0x8003 | Supervisor → Driver  | Registration rejected; error details       |
//! | `MSG_SERVICE_READY` | 0x8004 | Driver → Supervisor  | Service fully operational                  |
//! | `MSG_SERVICE_EXITING`| 0x8005| Driver → Supervisor  | Service shutting down cleanly              |

pub const MSG_BIND_READY: u16 = 0x8001;
pub const MSG_BIND_ASSIGNED: u16 = 0x8002;
/// Sent by the supervisor when a `MSG_BIND_READY` is rejected.
/// The driver MUST treat its VFS provider handle as invalid and either halt or retry.
pub const MSG_BIND_FAILED: u16 = 0x8003;
/// Sent by a driver/service after it has completed its internal initialization
/// and is ready to serve requests.  The supervisor records this for health monitoring.
pub const MSG_SERVICE_READY: u16 = 0x8004;
/// Sent by a driver/service just before it exits.  The supervisor uses this
/// to distinguish a clean shutdown from an unexpected crash.
pub const MSG_SERVICE_EXITING: u16 = 0x8005;

/// Error codes used in [`BindFailedPayload::error_code`].
pub mod errors {
    /// The `BIND_READY` message could not be parsed.
    pub const ERR_INVALID_MESSAGE: u32 = 1;
    /// No VFS provider handle was attached to the `BIND_READY` message.
    pub const ERR_NO_PROVIDER_HANDLE: u32 = 2;
    /// The `class_mask` in `BIND_READY` was zero or contained no recognised class bits.
    pub const ERR_UNKNOWN_CLASS: u32 = 3;
    /// The kernel `vfs_mount` call failed for the allocated path.
    pub const ERR_MOUNT_FAILED: u32 = 4;
}

/// Class bits for identifying the type of device being registered.
pub mod classes {
    pub const DISPLAY_CARD: u32 = 1 << 0;
    pub const FRAMEBUFFER: u32 = 1 << 1;
    pub const INPUT_EVENT: u32 = 1 << 2;
    pub const BLOCK_DEVICE: u32 = 1 << 3;
    pub const NETWORK_INTERFACE: u32 = 1 << 4;
    pub const SOUND_CARD: u32 = 1 << 5;
}

/// Payload for MSG_BIND_READY (Driver -> Supervisor)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BindReadyPayload {
    /// The unique token assigned to the driver by the supervisor at launch.
    pub bind_instance_id: u64,
    /// Bitmask of device classes provided by this driver (see `classes` mod).
    pub class_mask: u32,
    /// Reserved for future alignment/metadata.
    pub _reserved: u32,
}

/// Payload for MSG_BIND_ASSIGNED (Supervisor -> Driver)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BindAssignedPayload {
    /// Echoed bind_instance_id for confirmation.
    pub bind_instance_id: u64,
    /// Status code (0 for success, non-zero for error).
    pub status: u32,
    /// Canonical unit number assigned within the primary class.
    pub unit_number: u32,
    /// Canonical primary path assigned by the supervisor (e.g., "/dev/display/card0").
    pub primary_path: [u8; 64],
}

pub const BIND_READY_PAYLOAD_SIZE: usize = 16;
pub const BIND_ASSIGNED_PAYLOAD_SIZE: usize = 80; // 8 + 4 + 4 + 64

pub fn encode_bind_ready_le(payload: &BindReadyPayload, out: &mut [u8]) -> Option<usize> {
    if out.len() < BIND_READY_PAYLOAD_SIZE {
        return None;
    }
    out[0..8].copy_from_slice(&payload.bind_instance_id.to_le_bytes());
    out[8..12].copy_from_slice(&payload.class_mask.to_le_bytes());
    out[12..16].copy_from_slice(&payload._reserved.to_le_bytes());
    Some(BIND_READY_PAYLOAD_SIZE)
}

pub fn decode_bind_ready_le(buf: &[u8]) -> Option<BindReadyPayload> {
    if buf.len() < BIND_READY_PAYLOAD_SIZE {
        return None;
    }
    Some(BindReadyPayload {
        bind_instance_id: u64::from_le_bytes(buf[0..8].try_into().ok()?),
        class_mask: u32::from_le_bytes(buf[8..12].try_into().ok()?),
        _reserved: u32::from_le_bytes(buf[12..16].try_into().ok()?),
    })
}

pub fn encode_bind_assigned_le(payload: &BindAssignedPayload, out: &mut [u8]) -> Option<usize> {
    if out.len() < BIND_ASSIGNED_PAYLOAD_SIZE {
        return None;
    }
    out[0..8].copy_from_slice(&payload.bind_instance_id.to_le_bytes());
    out[8..12].copy_from_slice(&payload.status.to_le_bytes());
    out[12..16].copy_from_slice(&payload.unit_number.to_le_bytes());
    out[16..80].copy_from_slice(&payload.primary_path);
    Some(BIND_ASSIGNED_PAYLOAD_SIZE)
}

pub fn decode_bind_assigned_le(buf: &[u8]) -> Option<BindAssignedPayload> {
    if buf.len() < BIND_ASSIGNED_PAYLOAD_SIZE {
        return None;
    }
    let mut primary_path = [0u8; 64];
    primary_path.copy_from_slice(&buf[16..80]);
    Some(BindAssignedPayload {
        bind_instance_id: u64::from_le_bytes(buf[0..8].try_into().ok()?),
        status: u32::from_le_bytes(buf[8..12].try_into().ok()?),
        unit_number: u32::from_le_bytes(buf[12..16].try_into().ok()?),
        primary_path,
    })
}

/// Payload for MSG_BIND_FAILED (Supervisor -> Driver)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BindFailedPayload {
    /// Echoed bind_instance_id so the driver can correlate the failure.
    pub bind_instance_id: u64,
    /// Machine-readable error code (see [`errors`] module).
    pub error_code: u32,
    /// Reserved / padding.
    pub _reserved: u32,
    /// Human-readable reason string (UTF-8, NUL-padded).
    pub reason: [u8; 64],
}

/// Payload for MSG_SERVICE_READY (Driver -> Supervisor)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ServiceReadyPayload {
    /// The bind_instance_id that was confirmed in MSG_BIND_ASSIGNED.
    pub bind_instance_id: u64,
    /// Reserved / padding.
    pub _reserved: u64,
}

/// Payload for MSG_SERVICE_EXITING (Driver -> Supervisor)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ServiceExitingPayload {
    /// The bind_instance_id that was confirmed in MSG_BIND_ASSIGNED.
    pub bind_instance_id: u64,
    /// Exit/reason code: 0 = clean shutdown, non-zero = error.
    pub exit_code: u32,
    /// Reserved / padding.
    pub _reserved: u32,
}

pub const BIND_FAILED_PAYLOAD_SIZE: usize = 80; // 8 + 4 + 4 + 64
pub const SERVICE_READY_PAYLOAD_SIZE: usize = 16; // 8 + 8
pub const SERVICE_EXITING_PAYLOAD_SIZE: usize = 16; // 8 + 4 + 4

pub fn encode_bind_failed_le(payload: &BindFailedPayload, out: &mut [u8]) -> Option<usize> {
    if out.len() < BIND_FAILED_PAYLOAD_SIZE {
        return None;
    }
    out[0..8].copy_from_slice(&payload.bind_instance_id.to_le_bytes());
    out[8..12].copy_from_slice(&payload.error_code.to_le_bytes());
    out[12..16].copy_from_slice(&payload._reserved.to_le_bytes());
    out[16..80].copy_from_slice(&payload.reason);
    Some(BIND_FAILED_PAYLOAD_SIZE)
}

pub fn decode_bind_failed_le(buf: &[u8]) -> Option<BindFailedPayload> {
    if buf.len() < BIND_FAILED_PAYLOAD_SIZE {
        return None;
    }
    let mut reason = [0u8; 64];
    reason.copy_from_slice(&buf[16..80]);
    Some(BindFailedPayload {
        bind_instance_id: u64::from_le_bytes(buf[0..8].try_into().ok()?),
        error_code: u32::from_le_bytes(buf[8..12].try_into().ok()?),
        _reserved: u32::from_le_bytes(buf[12..16].try_into().ok()?),
        reason,
    })
}

pub fn encode_service_ready_le(payload: &ServiceReadyPayload, out: &mut [u8]) -> Option<usize> {
    if out.len() < SERVICE_READY_PAYLOAD_SIZE {
        return None;
    }
    out[0..8].copy_from_slice(&payload.bind_instance_id.to_le_bytes());
    out[8..16].copy_from_slice(&payload._reserved.to_le_bytes());
    Some(SERVICE_READY_PAYLOAD_SIZE)
}

pub fn decode_service_ready_le(buf: &[u8]) -> Option<ServiceReadyPayload> {
    if buf.len() < SERVICE_READY_PAYLOAD_SIZE {
        return None;
    }
    Some(ServiceReadyPayload {
        bind_instance_id: u64::from_le_bytes(buf[0..8].try_into().ok()?),
        _reserved: u64::from_le_bytes(buf[8..16].try_into().ok()?),
    })
}

pub fn encode_service_exiting_le(payload: &ServiceExitingPayload, out: &mut [u8]) -> Option<usize> {
    if out.len() < SERVICE_EXITING_PAYLOAD_SIZE {
        return None;
    }
    out[0..8].copy_from_slice(&payload.bind_instance_id.to_le_bytes());
    out[8..12].copy_from_slice(&payload.exit_code.to_le_bytes());
    out[12..16].copy_from_slice(&payload._reserved.to_le_bytes());
    Some(SERVICE_EXITING_PAYLOAD_SIZE)
}

pub fn decode_service_exiting_le(buf: &[u8]) -> Option<ServiceExitingPayload> {
    if buf.len() < SERVICE_EXITING_PAYLOAD_SIZE {
        return None;
    }
    Some(ServiceExitingPayload {
        bind_instance_id: u64::from_le_bytes(buf[0..8].try_into().ok()?),
        exit_code: u32::from_le_bytes(buf[8..12].try_into().ok()?),
        _reserved: u32::from_le_bytes(buf[12..16].try_into().ok()?),
    })
}
