//! Typed message delivery syscall handlers.
//!
//! Implements:
//! - `sys_msg_send`      — deliver one typed message directly to a process by PID
//! - `sys_msg_broadcast` — broadcast one typed message to all members of a pgid
//!
//! Both handlers share the same ABI layout:
//!
//! | arg | name        | description                           |
//! |-----|-------------|---------------------------------------|
//! | 0   | target      | PID (send) or PGID (broadcast)        |
//! | 1   | kind_id_ptr | user pointer to 16-byte KindId        |
//! | 2   | payload_ptr | user pointer to payload bytes         |
//! | 3   | payload_len | byte length of payload                |
//!
//! ## Membership semantics
//!
//! `sys_msg_broadcast` takes a **snapshot** of the process-group membership
//! at call time.  Any processes that join or leave the group after the snapshot
//! is taken are unaffected.  This is the documented behavior for the prototype.
//!
//! ## Partial failure semantics
//!
//! `sys_msg_broadcast` continues fanout past per-recipient failures.  The
//! return value encodes both counts:
//!   bits 31:16 – failures (saturated to 0xFFFF)
//!   bits 15:0  – successes (saturated to 0xFFFF)
//!
//! Top-level errors (`EINVAL`, `EFAULT`) are still returned as negative errno.

use super::copyin;
use crate::message::KindId;
use crate::message::delivery::{
    DeliveryFailureReason, GroupBroadcastError, deliver_typed_to_process,
    broadcast_typed_to_group_snapshot,
};
use crate::syscall::validate::validate_user_range;
use abi::errors::{Errno, SysResult};
use alloc::vec;

/// Maximum payload accepted through the syscall boundary.
///
/// This cap prevents a single large message from monopolising kernel heap.
const MAX_PAYLOAD_LEN: usize = 65536;

/// Size of a `KindId` in bytes (UUID-style 128-bit value).
const KIND_ID_LEN: usize = 16;

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Copy `KindId` bytes and payload from user space.
///
/// Returns `(KindId, payload)` or an `Errno` on any fault / size violation.
fn copyin_message(
    kind_id_ptr: usize,
    payload_ptr: usize,
    payload_len: usize,
) -> SysResult<(crate::message::Message,)> {
    // Reject oversized payloads early.
    if payload_len > MAX_PAYLOAD_LEN {
        return Err(Errno::EINVAL);
    }

    // Validate and copy the 16-byte KindId.
    validate_user_range(kind_id_ptr, KIND_ID_LEN, false)?;
    let mut kind_bytes = [0u8; KIND_ID_LEN];
    unsafe { copyin(&mut kind_bytes, kind_id_ptr)? };
    let kind = KindId(kind_bytes);

    // Validate and copy the payload (empty payload is valid).
    let payload = if payload_len > 0 {
        validate_user_range(payload_ptr, payload_len, false)?;
        let mut buf = vec![0u8; payload_len];
        unsafe { copyin(&mut buf, payload_ptr)? };
        buf
    } else {
        alloc::vec![]
    };

    let message = crate::message::Message::new(kind, payload);
    Ok((message,))
}

// ── Public syscall handlers ───────────────────────────────────────────────────

/// `sys_msg_send(pid, kind_id_ptr, payload_ptr, payload_len)`
///
/// Deliver one typed message directly to process `pid`.
///
/// Returns:
/// - `0` on success
/// - `-ESRCH`  if the recipient process is not found
/// - `-EAGAIN` if the recipient inbox is full
/// - `-EINVAL` if arguments are malformed
/// - `-EFAULT` if user pointers are invalid
pub fn sys_msg_send(
    pid: usize,
    kind_id_ptr: usize,
    payload_ptr: usize,
    payload_len: usize,
) -> SysResult<usize> {
    let pid = pid as u32;
    let (message,) = copyin_message(kind_id_ptr, payload_ptr, payload_len)?;

    deliver_typed_to_process(pid, &message).map_err(|e| match e {
        DeliveryFailureReason::RecipientExited => Errno::ESRCH,
        DeliveryFailureReason::InboxFull => Errno::EAGAIN,
    })?;

    Ok(0)
}

/// `sys_msg_broadcast(pgid, kind_id_ptr, payload_ptr, payload_len)`
///
/// Broadcast one typed message to all current members of process group `pgid`.
///
/// Membership is snapshotted once at call time; fanout continues past
/// per-recipient failures.
///
/// Return value (on success):
/// - `bits 31:16` — number of failures (saturated to `0xFFFF`)
/// - `bits 15:0`  — number of successes (saturated to `0xFFFF`)
///
/// Returns negative errno for top-level errors:
/// - `-EINVAL` if `pgid` == 0 or pointers are malformed
/// - `-EFAULT` if user pointers are invalid
pub fn sys_msg_broadcast(
    pgid: usize,
    kind_id_ptr: usize,
    payload_ptr: usize,
    payload_len: usize,
) -> SysResult<usize> {
    let pgid = pgid as u32;
    let (message,) = copyin_message(kind_id_ptr, payload_ptr, payload_len)?;

    let report =
        broadcast_typed_to_group_snapshot(pgid, &message).map_err(|e| match e {
            GroupBroadcastError::InvalidGroupId => Errno::EINVAL,
        })?;

    // Pack (failures, successes) into the usize return value.
    let successes = report.succeeded.min(0xFFFF) as usize;
    let failures = report.failed.min(0xFFFF) as usize;
    Ok((failures << 16) | successes)
}
