//! Job exit notification: emit `JobExit` via the canonical Inbox delivery path.
//!
//! # Design
//!
//! This module is the **single emission point** for job-exit notifications
//! carried through the `Message/Inbox` substrate.  It wires together:
//!
//! * canonical `JobExit` payload construction (via `thingos::job`)
//! * `Message` envelope wrapping (via `kernel::message::bridge`)
//! * `Inbox` delivery (via `kernel::inbox`)
//!
//! No side channels or bespoke callbacks are used.  Delivery failure is
//! logged and observable but never corrupts lifecycle state.
//!
//! # Emission rules
//!
//! * Exactly one `JobExit` message per job exit event.  Duplicate suppression
//!   is the caller's responsibility (the scheduler calls this once, at the
//!   thread-group-leader exit point).
//! * If no observer inbox is registered for the job (`None`), the exit
//!   completes normally and no message is emitted.
//! * Delivery failure (inbox not found, inbox full, inbox closed) is logged
//!   via `kdebug!` and does **not** panic or corrupt lifecycle state.
//!
//! # Follow-on work
//!
//! * `register_exit_observer` is the only registration path today; future
//!   phases will add group/supervisor observers.
//! * Payload encoding (`JobExit::encode_as_notification`) is defined in
//!   `thingos::job` and is the canonical wire format.

use crate::inbox::{get_inbox, InboxId, MessageEnvelope};
use crate::message::bridge::message_from_parts;
use crate::message::KindId;
use thingos::job::{JobExit, JobState};

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// Emit a canonical `JobExit` notification to the registered observer inbox.
///
/// Constructs a [`JobExit`] value from `exit_code`, serialises it together
/// with `job_id` using [`JobExit::encode_as_notification`], wraps the result
/// in a [`Message`] with [`KindId::THINGOS_JOB_EXIT`], and enqueues it in
/// the [`Inbox`] identified by `inbox_id`.
///
/// This is the **single production point** for job-exit messaging and must
/// be called at most once per job exit event.
///
/// # Failure behaviour
///
/// * If the inbox is not found (already closed/dropped): event is logged and
///   the function returns.  Lifecycle state is not affected.
/// * If the inbox is full: delivery failure is logged.  Lifecycle state is
///   not affected.
/// * If the inbox is closed: delivery failure is logged.  Lifecycle state
///   is not affected.
pub fn emit_job_exit(inbox_id: InboxId, job_id: u32, exit_code: i32) {
    let job_exit = JobExit {
        state: JobState::Exited,
        code: Some(exit_code),
    };
    let payload = job_exit.encode_as_notification(job_id);
    let message = message_from_parts(KindId::THINGOS_JOB_EXIT, payload);
    let envelope = MessageEnvelope::anonymous(message);

    let Some(inbox) = get_inbox(inbox_id) else {
        crate::kdebug!(
            "job::notify: observer inbox not found job_id={} inbox_id={:?}; delivery skipped",
            job_id,
            inbox_id
        );
        return;
    };

    match inbox.send(envelope) {
        Ok(()) => {
            crate::kdebug!(
                "job::notify: JobExit delivered job_id={} inbox_id={:?}",
                job_id,
                inbox_id
            );
        }
        Err(err) => {
            crate::kdebug!(
                "job::notify: JobExit delivery failed job_id={} inbox_id={:?} err={:?}",
                job_id,
                inbox_id,
                err
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Observer registration
// ---------------------------------------------------------------------------

/// Register an observer inbox to receive a `JobExit` notification for `pid`.
///
/// When the job (process) identified by `pid` exits, a `JobExit` message will
/// be delivered to `inbox_id` via [`emit_job_exit`].
///
/// Returns `true` if the observer was registered successfully, or `false` if
/// no live process with the given `pid` was found.
///
/// Replaces any previously registered observer for this job.
pub fn register_exit_observer(pid: u32, inbox_id: InboxId) -> bool {
    let Some(pinfo_arc) = crate::sched::process_info_for_tid_current(pid as u64) else {
        return false;
    };
    let mut pi = pinfo_arc.lock();
    pi.lifecycle.exit_observer_inbox = Some(inbox_id);
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inbox::{create_inbox, DEFAULT_INBOX_CAPACITY};
    use crate::message::KindId;
    use thingos::job::JobExit;

    // ── emit_job_exit: basic delivery ────────────────────────────────────────

    /// Emitting a JobExit notification enqueues exactly one message.
    #[test]
    fn emit_job_exit_delivers_one_message() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        emit_job_exit(inbox_id, 42, 0);

        let inbox = get_inbox(inbox_id).expect("inbox must exist");
        let envelope = inbox.try_recv().expect("no recv error").expect("should have one message");
        assert!(inbox.try_recv().ok().flatten().is_none(), "should have exactly one message");

        // Clean up.
        crate::inbox::close_inbox(inbox_id);
        let _ = envelope;
    }

    /// The delivered message carries the THINGOS_JOB_EXIT kind.
    #[test]
    fn emit_job_exit_message_has_correct_kind() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        emit_job_exit(inbox_id, 7, 1);

        let inbox = get_inbox(inbox_id).unwrap();
        let envelope = inbox.try_recv().expect("no recv error").expect("should have a message");
        assert_eq!(envelope.message.kind, KindId::THINGOS_JOB_EXIT);

        crate::inbox::close_inbox(inbox_id);
    }

    /// The payload decodes to the correct job_id and exit code.
    #[test]
    fn emit_job_exit_payload_decodes_correctly() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        let job_id = 99u32;
        let code = 42i32;
        emit_job_exit(inbox_id, job_id, code);

        let inbox = get_inbox(inbox_id).unwrap();
        let envelope = inbox.try_recv().expect("no recv error").expect("should have a message");
        let (decoded_id, decoded_exit) =
            JobExit::decode_notification(&envelope.message.payload).expect("should decode");
        assert_eq!(decoded_id, job_id);
        assert_eq!(decoded_exit.state, JobState::Exited);
        assert_eq!(decoded_exit.code, Some(code));

        crate::inbox::close_inbox(inbox_id);
    }

    /// Negative exit codes are preserved in the notification payload.
    #[test]
    fn emit_job_exit_negative_code_preserved() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        emit_job_exit(inbox_id, 5, -1);

        let inbox = get_inbox(inbox_id).unwrap();
        let envelope = inbox.try_recv().expect("no recv error").expect("should have a message");
        let (_, decoded) =
            JobExit::decode_notification(&envelope.message.payload).expect("should decode");
        assert_eq!(decoded.code, Some(-1));

        crate::inbox::close_inbox(inbox_id);
    }

    /// No sender TID is set on the notification envelope (system event).
    #[test]
    fn emit_job_exit_envelope_has_no_sender() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        emit_job_exit(inbox_id, 3, 0);

        let inbox = get_inbox(inbox_id).unwrap();
        let envelope = inbox.try_recv().expect("no recv error").expect("should have a message");
        assert_eq!(envelope.sender, None);

        crate::inbox::close_inbox(inbox_id);
    }

    // ── emit_job_exit: inbox unavailable ────────────────────────────────────

    /// Emit to a non-existent InboxId does not panic.
    #[test]
    fn emit_job_exit_no_inbox_does_not_panic() {
        // Use an arbitrary InboxId that is very unlikely to be live.
        let phantom_id = InboxId(0xdead_beef_cafe_babe);
        // Must not panic.
        emit_job_exit(phantom_id, 1, 0);
    }

    /// Emit to a closed inbox does not panic.
    #[test]
    fn emit_job_exit_closed_inbox_does_not_panic() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        crate::inbox::close_inbox(inbox_id);
        // Must not panic.
        emit_job_exit(inbox_id, 1, 0);
    }

    /// When the inbox is full, emit does not panic.
    #[test]
    fn emit_job_exit_full_inbox_does_not_panic() {
        // Create a capacity-1 inbox.
        let inbox_id = create_inbox(1);
        // Fill it.
        emit_job_exit(inbox_id, 10, 0);
        // Attempt a second emit — inbox is full; must not panic.
        emit_job_exit(inbox_id, 10, 0);

        crate::inbox::close_inbox(inbox_id);
    }

    // ── emit_job_exit: single-delivery guarantee ─────────────────────────────

    /// Multiple distinct job exits produce distinct notifications.
    #[test]
    fn multiple_jobs_produce_distinct_notifications() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        emit_job_exit(inbox_id, 100, 0);
        emit_job_exit(inbox_id, 200, 1);
        emit_job_exit(inbox_id, 300, 2);

        let inbox = get_inbox(inbox_id).unwrap();

        let env0 = inbox.try_recv().expect("no recv error").expect("should have a message");
        let (id0, exit0) =
            JobExit::decode_notification(&env0.message.payload).unwrap();

        let env1 = inbox.try_recv().expect("no recv error").expect("should have a message");
        let (id1, exit1) =
            JobExit::decode_notification(&env1.message.payload).unwrap();

        let env2 = inbox.try_recv().expect("no recv error").expect("should have a message");
        let (id2, exit2) =
            JobExit::decode_notification(&env2.message.payload).unwrap();

        assert!(inbox.try_recv().ok().flatten().is_none());

        assert_eq!(id0, 100);
        assert_eq!(exit0.code, Some(0));
        assert_eq!(id1, 200);
        assert_eq!(exit1.code, Some(1));
        assert_eq!(id2, 300);
        assert_eq!(exit2.code, Some(2));

        crate::inbox::close_inbox(inbox_id);
    }

    // ── payload correctness ──────────────────────────────────────────────────

    /// The kind constant in the message matches THINGOS_JOB_EXIT.
    #[test]
    fn message_kind_matches_thingos_job_exit_constant() {
        let inbox_id = create_inbox(DEFAULT_INBOX_CAPACITY);
        emit_job_exit(inbox_id, 1, 0);

        let inbox = get_inbox(inbox_id).unwrap();
        let envelope = inbox.try_recv().expect("no recv error").expect("should have a message");
        assert_eq!(
            envelope.message.kind.0,
            [0xc2, 0x60, 0x8e, 0x30, 0xff, 0xa2, 0xa2, 0xda,
             0x8b, 0x96, 0x22, 0x8d, 0x3e, 0xd0, 0x11, 0x74],
            "kind bytes must match KIND_ID_THINGOS_JOB_EXIT"
        );

        crate::inbox::close_inbox(inbox_id);
    }

    // ── lifecycle coherence ──────────────────────────────────────────────────

    /// Delivery failure (full inbox) does not affect the exit code recorded in
    /// the message — the payload is still correctly formed even when the send
    /// is rejected.
    #[test]
    fn payload_is_well_formed_even_when_inbox_full() {
        // Use capacity 1; fill with a known message, then attempt a second.
        let inbox_id = create_inbox(1);
        emit_job_exit(inbox_id, 11, 5);   // succeeds
        emit_job_exit(inbox_id, 22, 99);  // fails (full) but must not corrupt state

        let inbox = get_inbox(inbox_id).unwrap();
        let env = inbox.try_recv().expect("no recv error").expect("should have a message");
        let (id, exit) = JobExit::decode_notification(&env.message.payload).unwrap();
        assert_eq!(id, 11);
        assert_eq!(exit.code, Some(5));
        assert!(inbox.try_recv().ok().flatten().is_none(), "only the first message should be present");

        crate::inbox::close_inbox(inbox_id);
    }
}
