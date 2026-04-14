//! Typed message delivery prototype: direct send + process-group broadcast.
//!
//! This module prototypes group broadcast as a delivery strategy layered on top
//! of ordinary per-recipient inbox enqueue.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::message::Message;
use crate::task::{
    MessageDeliveryKind, MessageEnqueueError, ProcessMessage, ProcessMessageMetadata,
};

/// Error returned when a broadcast target is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupBroadcastError {
    /// Group ID 0 is reserved/invalid in the transitional process-group model.
    InvalidGroupId,
}

/// Per-recipient delivery failure reason.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryFailureReason {
    /// Recipient process could not be found at delivery time.
    RecipientExited,
    /// Recipient inbox is full.
    InboxFull,
}

/// One failed recipient result inside a group broadcast report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecipientFailure {
    pub pid: u32,
    pub reason: DeliveryFailureReason,
}

/// Aggregate result of one group fanout attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupBroadcastReport {
    /// Number of members in the send-time membership snapshot.
    pub targeted: usize,
    /// Number of successful per-recipient enqueues.
    pub succeeded: usize,
    /// Number of failed per-recipient enqueues.
    pub failed: usize,
    /// Detailed per-recipient failure list.
    pub failures: Vec<RecipientFailure>,
}

impl GroupBroadcastReport {
    fn from_failures(targeted: usize, failures: Vec<RecipientFailure>) -> Self {
        let failed = failures.len();
        let succeeded = targeted.saturating_sub(failed);
        Self {
            targeted,
            succeeded,
            failed,
            failures,
        }
    }
}

/// Deliver one typed message directly to `pid`.
///
/// Uses the same per-recipient enqueue path as group fanout.
pub fn deliver_typed_to_process(pid: u32, message: &Message) -> Result<(), DeliveryFailureReason> {
    let sender = sender_context();
    let metadata = ProcessMessageMetadata {
        sender_tid: sender.sender_tid,
        sender_job: sender.sender_job,
        target_group: None,
        delivery_kind: MessageDeliveryKind::Direct,
        broadcast_sequence: None,
    };
    enqueue_to_process(pid, message.clone(), metadata)
}

/// Broadcast one typed message to all members of process-group `pgid`.
///
/// Membership is snapshotted once at send time and fanout is then performed
/// over the snapshot, preserving deterministic delivery counts.
pub fn broadcast_typed_to_group_snapshot(
    pgid: u32,
    message: &Message,
) -> Result<GroupBroadcastReport, GroupBroadcastError> {
    if pgid == 0 {
        return Err(GroupBroadcastError::InvalidGroupId);
    }

    let recipients = snapshot_group_members_by_pgid(pgid);
    let sender = sender_context();

    let report = fanout_snapshot(
        &recipients,
        |idx| ProcessMessageMetadata {
            sender_tid: sender.sender_tid,
            sender_job: sender.sender_job,
            target_group: Some(pgid),
            delivery_kind: MessageDeliveryKind::GroupBroadcast,
            broadcast_sequence: Some(idx as u64),
        },
        |pid, metadata| enqueue_to_process(pid, message.clone(), metadata),
    );

    crate::kdebug!(
        "message::broadcast pgid={} targeted={} ok={} failed={}",
        pgid,
        report.targeted,
        report.succeeded,
        report.failed
    );
    for failure in &report.failures {
        crate::kdebug!(
            "message::broadcast pgid={} pid={} failure={:?}",
            pgid,
            failure.pid,
            failure.reason
        );
    }

    Ok(report)
}

#[derive(Clone, Copy)]
struct SenderContext {
    sender_tid: u64,
    sender_job: Option<u32>,
}

fn sender_context() -> SenderContext {
    let sender_tid = unsafe { crate::sched::current_tid_current() };
    let sender_job = crate::sched::process_info_current().map(|p| p.lock().pid);
    SenderContext {
        sender_tid,
        sender_job,
    }
}

fn enqueue_to_process(
    pid: u32,
    message: Message,
    metadata: ProcessMessageMetadata,
) -> Result<(), DeliveryFailureReason> {
    let Some(pinfo) = crate::sched::process_info_for_tid_current(pid as u64) else {
        return Err(DeliveryFailureReason::RecipientExited);
    };

    let mut process = pinfo.lock();
    process
        .unix_compat
        .enqueue_message(ProcessMessage { message, metadata })
        .map_err(|err| match err {
            MessageEnqueueError::InboxFull { .. } => DeliveryFailureReason::InboxFull,
        })?;
    let tids = process.lifecycle.thread_ids.clone();
    drop(process);

    for tid in tids {
        unsafe { crate::sched::wake_task_erased(tid as u64) };
    }

    Ok(())
}

fn snapshot_group_members_by_pgid(pgid: u32) -> Vec<u32> {
    let mut seen = BTreeSet::new();
    let mut members = Vec::new();

    for snapshot in crate::sched::list_processes_current() {
        if !seen.insert(snapshot.pid) {
            continue;
        }

        let Some(pinfo) = crate::sched::process_info_for_tid_current(snapshot.pid as u64) else {
            continue;
        };

        let process = pinfo.lock();
        if process.unix_compat.pgid == pgid {
            members.push(process.pid);
        }
    }

    members
}

fn fanout_snapshot(
    recipients: &[u32],
    metadata_for_index: impl Fn(usize) -> ProcessMessageMetadata,
    mut deliver: impl FnMut(u32, ProcessMessageMetadata) -> Result<(), DeliveryFailureReason>,
) -> GroupBroadcastReport {
    let mut failures = Vec::new();

    for (idx, pid) in recipients.iter().copied().enumerate() {
        let metadata = metadata_for_index(idx);
        if let Err(reason) = deliver(pid, metadata) {
            failures.push(RecipientFailure { pid, reason });
        }
    }

    GroupBroadcastReport::from_failures(recipients.len(), failures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::KindId;

    fn make_message() -> Message {
        Message::new(KindId::THINGOS_MESSAGE, alloc::vec![1, 2, 3])
    }

    fn make_meta(idx: usize, group: u32) -> ProcessMessageMetadata {
        ProcessMessageMetadata {
            sender_tid: 7,
            sender_job: Some(42),
            target_group: Some(group),
            delivery_kind: MessageDeliveryKind::GroupBroadcast,
            broadcast_sequence: Some(idx as u64),
        }
    }

    #[test]
    fn fanout_basic_all_success() {
        let recipients = [11u32, 12, 13, 14];
        let mut delivered = Vec::new();

        let report = fanout_snapshot(
            &recipients,
            |idx| make_meta(idx, 55),
            |pid, metadata| {
                delivered.push((pid, metadata));
                Ok(())
            },
        );

        assert_eq!(report.targeted, 4);
        assert_eq!(report.succeeded, 4);
        assert_eq!(report.failed, 0);
        assert!(report.failures.is_empty());
        assert_eq!(delivered.len(), 4);
        assert_eq!(delivered[0].0, 11);
        assert_eq!(delivered[3].0, 14);
    }

    #[test]
    fn fanout_empty_group_is_clean_success() {
        let recipients: [u32; 0] = [];
        let report = fanout_snapshot(&recipients, |idx| make_meta(idx, 2), |_pid, _metadata| Ok(()));

        assert_eq!(report.targeted, 0);
        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 0);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn sender_included_semantics_are_member_driven() {
        let sender_pid = 77u32;
        let recipients = [sender_pid, 78u32];
        let mut seen = Vec::new();

        let report = fanout_snapshot(
            &recipients,
            |idx| make_meta(idx, 9),
            |pid, _metadata| {
                seen.push(pid);
                Ok(())
            },
        );

        assert_eq!(report.failed, 0);
        assert_eq!(seen, alloc::vec![77u32, 78u32]);
    }

    #[test]
    fn one_recipient_failure_does_not_block_others() {
        let recipients = [200u32, 201u32, 202u32];
        let mut successes = Vec::new();

        let report = fanout_snapshot(
            &recipients,
            |idx| make_meta(idx, 33),
            |pid, _metadata| {
                if pid == 201 {
                    return Err(DeliveryFailureReason::InboxFull);
                }
                successes.push(pid);
                Ok(())
            },
        );

        assert_eq!(report.targeted, 3);
        assert_eq!(report.succeeded, 2);
        assert_eq!(report.failed, 1);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].pid, 201);
        assert_eq!(report.failures[0].reason, DeliveryFailureReason::InboxFull);
        assert_eq!(successes, alloc::vec![200u32, 202u32]);
    }

    #[test]
    fn fanout_preserves_message_and_metadata_shape() {
        let recipients = [401u32, 402u32];
        let original = make_message();
        let mut delivered = Vec::new();

        let report = fanout_snapshot(
            &recipients,
            |idx| make_meta(idx, 88),
            |pid, metadata| {
                delivered.push((pid, original.clone(), metadata));
                Ok(())
            },
        );

        assert_eq!(report.failed, 0);
        assert_eq!(delivered.len(), 2);
        assert_eq!(delivered[0].1, original);
        assert_eq!(delivered[0].2.target_group, Some(88));
        assert_eq!(delivered[0].2.delivery_kind, MessageDeliveryKind::GroupBroadcast);
        assert_eq!(delivered[1].2.broadcast_sequence, Some(1));
    }

    #[test]
    fn fanout_uses_snapshot_even_if_live_membership_changes() {
        let recipients_snapshot = alloc::vec![501u32, 502u32];
        let mut live_membership = alloc::vec![501u32, 502u32, 503u32];
        let mut delivered = Vec::new();

        let report = fanout_snapshot(
            &recipients_snapshot,
            |idx| make_meta(idx, 12),
            |pid, _metadata| {
                // Simulate concurrent mutation to live membership while fanout
                // is in progress. Snapshot recipients remain authoritative.
                if pid == 501 {
                    live_membership.push(999);
                }
                delivered.push(pid);
                Ok(())
            },
        );

        assert_eq!(report.targeted, 2);
        assert_eq!(report.succeeded, 2);
        assert_eq!(delivered, alloc::vec![501u32, 502u32]);
        assert!(live_membership.contains(&503u32));
        assert!(live_membership.contains(&999u32));
    }
}
