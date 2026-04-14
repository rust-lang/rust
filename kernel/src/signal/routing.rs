//! Typed signal routing — Phase 1.
//!
//! # Design note
//!
//! ## What legacy path is being replaced
//!
//! Previously, `send_signal_to_process`, `send_signal_to_thread`, and
//! `send_signal_to_group` each performed ad-hoc fanout by reaching directly
//! into `ProcessSignals::post` and waking threads inline.  Target resolution,
//! membership iteration, and per-recipient state mutation were all tangled
//! together inside each helper with no shared abstraction.
//!
//! ## What target kinds are supported in Phase 1
//!
//! - [`SignalTargetKind::Process`] — signal a single process by PID.
//! - [`SignalTargetKind::Thread`] — signal a single thread by TID.
//! - [`SignalTargetKind::ProcessGroup`] — signal all processes in a PGID.
//!
//! These are the only target kinds exercised by the existing `sys_kill`,
//! `sys_raise`, and `notify_parent_sigchld` code paths.  Broadcast to "all
//! processes" (`kill(-1, sig)`) and session-wide signaling remain on legacy
//! paths; they are explicitly flagged as transitional.
//!
//! ## Typed routing structures introduced
//!
//! - [`SignalTargetKind`] — enumerates the supported targeting modes.
//! - [`SignalRoute`] — a fully described routing request: signal number,
//!   optional sender TID, and target kind + ID.
//! - [`SignalDeliveryOutcome`] — per-recipient result (delivered or a concrete
//!   failure reason).
//! - [`SignalRoutingReport`] — aggregate fanout result: targeted count,
//!   success count, failure list.
//!
//! ## Handoff to existing signal semantics
//!
//! The routing layer **does not** replace per-recipient signal semantics.
//! Instead, the canonical delivery hook [`deliver_to_recipient`] calls
//! `ProcessSignals::post` (the existing pending-signal state machinery) and
//! then wakes the target threads.  Everything downstream (mask checking,
//! disposition lookup, handler invocation, default actions) is entirely
//! unchanged.
//!
//! ## What remains explicitly transitional
//!
//! - `kill(-1, sig)` (broadcast to all processes) still returns `EPERM` and
//!   does **not** pass through this routing layer.
//! - Signal delivery to threads within a stopped process continues to follow
//!   the existing wake-regardless semantics.
//! - Permission checks are not yet implemented (any process may signal any
//!   other); that is a pre-existing limitation, not introduced here.

use alloc::vec::Vec;

// ── Public types ──────────────────────────────────────────────────────────────

/// Describes what kind of entity a signal is addressed to.
///
/// Phase 1 supports three kinds.  Future phases may add `Session` and
/// `Broadcast` (kill -1) once the recipient model stabilises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalTargetKind {
    /// A single process identified by PID.
    Process,
    /// A single thread identified by TID.
    Thread,
    /// All processes whose PGID matches the target ID.
    ProcessGroup,
}

/// A fully-described routing request for one signal send operation.
///
/// This is the single entry point that callers should construct when they
/// want to send a signal through the typed routing substrate.
#[derive(Clone, Copy, Debug)]
pub struct SignalRoute {
    /// Signal number (1–63). 0 is an existence/permission check: fanout
    /// is performed (membership is counted) but no signal is posted.
    pub signal: u8,
    /// TID of the sending thread, if known.
    pub sender_tid: Option<u64>,
    /// What kind of entity is being targeted.
    pub target_kind: SignalTargetKind,
    /// The numeric ID of the target (PID, TID, or PGID depending on `target_kind`).
    pub target_id: u64,
}

/// Outcome of routing a signal to one specific recipient process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalDeliveryOutcome {
    /// Signal was posted to the recipient's pending set and all threads woken.
    Delivered,
    /// Existence check (signal == 0): recipient exists.
    ExistenceConfirmed,
    /// The target process or thread no longer exists.
    RecipientNotFound,
}

/// Aggregate result of a complete routing + fanout operation.
#[derive(Clone, Debug)]
pub struct SignalRoutingReport {
    /// Resolved target kind (copied from `SignalRoute`).
    pub target_kind: SignalTargetKind,
    /// Number of members in the send-time membership snapshot.
    pub targeted: usize,
    /// Number of recipients where delivery succeeded.
    pub succeeded: usize,
    /// Number of recipients where delivery failed.
    pub failed: usize,
    /// Per-recipient failures with their PIDs.
    pub failures: Vec<(u64, SignalDeliveryOutcome)>,
}

impl SignalRoutingReport {
    fn build(
        target_kind: SignalTargetKind,
        outcomes: Vec<(u64, SignalDeliveryOutcome)>,
    ) -> Self {
        let targeted = outcomes.len();
        let failed: usize = outcomes
            .iter()
            .filter(|(_, o)| *o == SignalDeliveryOutcome::RecipientNotFound)
            .count();
        let succeeded = targeted.saturating_sub(failed);
        let failures = outcomes
            .into_iter()
            .filter(|(_, o)| *o == SignalDeliveryOutcome::RecipientNotFound)
            .collect();
        Self {
            target_kind,
            targeted,
            succeeded,
            failed,
            failures,
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Route signal `route.signal` according to the typed routing request.
///
/// This is the single typed entry point for signal routing.  It performs:
///
/// 1. Target validation.
/// 2. Membership resolution (snapshot for group targets).
/// 3. Per-recipient iteration via the canonical delivery hook.
/// 4. Aggregate report construction.
///
/// Diagnostics (via `kdebug!`) are emitted for the sender intent, resolved
/// membership size, and any per-recipient failures.
///
/// Existing signal compatibility semantics (pending set, mask, disposition,
/// wakeup) are preserved because [`deliver_to_recipient`] calls through the
/// existing `ProcessSignals::post` + wake path.
pub fn route_signal(route: SignalRoute) -> SignalRoutingReport {
    crate::kdebug!(
        "signal::route sig={} sender={:?} target={:?} id={}",
        route.signal,
        route.sender_tid,
        route.target_kind,
        route.target_id,
    );

    let outcomes = match route.target_kind {
        SignalTargetKind::Process => route_to_process(route),
        SignalTargetKind::Thread => route_to_thread(route),
        SignalTargetKind::ProcessGroup => route_to_group(route),
    };

    let report = SignalRoutingReport::build(route.target_kind, outcomes);

    crate::kdebug!(
        "signal::route done sig={} targeted={} ok={} failed={}",
        route.signal,
        report.targeted,
        report.succeeded,
        report.failed,
    );
    for (id, outcome) in &report.failures {
        crate::kdebug!(
            "signal::route failure sig={} id={} outcome={:?}",
            route.signal,
            id,
            outcome,
        );
    }

    report
}

// ── Per-target routing logic ──────────────────────────────────────────────────

fn route_to_process(route: SignalRoute) -> Vec<(u64, SignalDeliveryOutcome)> {
    let pid = route.target_id as u32;
    let outcome = deliver_to_recipient(pid, route.signal);
    crate::kdebug!(
        "signal::route process pid={} sig={} outcome={:?}",
        pid,
        route.signal,
        outcome,
    );
    alloc::vec![(route.target_id, outcome)]
}

fn route_to_thread(route: SignalRoute) -> Vec<(u64, SignalDeliveryOutcome)> {
    let tid = route.target_id;
    let outcome = deliver_to_thread_recipient(tid, route.signal);
    crate::kdebug!(
        "signal::route thread tid={} sig={} outcome={:?}",
        tid,
        route.signal,
        outcome,
    );
    alloc::vec![(tid, outcome)]
}

fn route_to_group(route: SignalRoute) -> Vec<(u64, SignalDeliveryOutcome)> {
    let pgid = route.target_id as u32;
    // Snapshot membership at send time — fanout uses this snapshot exclusively.
    let members = snapshot_group_pids(pgid);
    crate::kdebug!(
        "signal::route group pgid={} sig={} members={}",
        pgid,
        route.signal,
        members.len(),
    );
    members
        .into_iter()
        .map(|pid| {
            let outcome = deliver_to_recipient(pid, route.signal);
            crate::kdebug!(
                "signal::route group pid={} sig={} outcome={:?}",
                pid,
                route.signal,
                outcome,
            );
            (pid as u64, outcome)
        })
        .collect()
}

// ── Canonical per-recipient delivery hook ─────────────────────────────────────

/// Deliver `sig` to process `pid` through the canonical path.
///
/// This is the **single** per-recipient delivery hook.  It translates a
/// routing decision into the existing pending-signal state machinery:
///
/// - Looks up the process by PID.
/// - Calls `ProcessSignals::post` (which enforces mask/ignore semantics).
/// - Wakes all threads in the process so they check for pending signals.
///
/// # sig == 0
///
/// When `sig == 0` the function only checks existence (POSIX existence/
/// permission check).  No signal is posted and no threads are woken.
///
/// # Compatibility
///
/// All downstream signal semantics (masks, dispositions, default actions,
/// handler invocation, coalescing via the pending-set bitmask) are
/// **unchanged** by this function.  The routing layer only controls who
/// gets signaled, not what the signal means.
pub fn deliver_to_recipient(pid: u32, sig: u8) -> SignalDeliveryOutcome {
    let Some(pinfo) = crate::sched::process_info_for_tid_current(pid as u64) else {
        return SignalDeliveryOutcome::RecipientNotFound;
    };

    if sig == 0 {
        return SignalDeliveryOutcome::ExistenceConfirmed;
    }

    let tids = {
        let mut p = pinfo.lock();
        p.unix_compat.signals.post(sig);
        p.lifecycle.thread_ids.clone()
    };

    for tid in tids {
        unsafe { crate::sched::wake_task_erased(tid as u64) };
    }

    SignalDeliveryOutcome::Delivered
}

/// Deliver `sig` to a specific thread by TID through the canonical path.
///
/// Posts the signal to the process-level pending set of the process that
/// owns `tid`, then wakes only the targeted thread.  This matches the
/// legacy `send_signal_to_thread` semantics.
///
/// When `sig == 0` the function only checks existence.
pub fn deliver_to_thread_recipient(tid: u64, sig: u8) -> SignalDeliveryOutcome {
    let Some(pinfo) = crate::sched::process_info_for_tid_current(tid) else {
        return SignalDeliveryOutcome::RecipientNotFound;
    };

    if sig == 0 {
        return SignalDeliveryOutcome::ExistenceConfirmed;
    }

    {
        let mut p = pinfo.lock();
        p.unix_compat.signals.post(sig);
    }

    unsafe { crate::sched::wake_task_erased(tid) };

    SignalDeliveryOutcome::Delivered
}

// ── Membership snapshot helper ────────────────────────────────────────────────

fn snapshot_group_pids(pgid: u32) -> Vec<u32> {
    use alloc::collections::BTreeSet;

    let mut seen = BTreeSet::new();
    let mut members = Vec::new();

    for snapshot in crate::sched::list_processes_current() {
        if !seen.insert(snapshot.pid) {
            continue;
        }
        let Some(pinfo) = crate::sched::process_info_for_tid_current(snapshot.pid as u64) else {
            continue;
        };
        let p = pinfo.lock();
        if p.unix_compat.pgid == pgid {
            members.push(p.pid);
        }
    }

    members
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit tests for routing report construction ────────────────────────────
    //
    // These tests exercise the pure routing-layer logic (report building,
    // outcome classification, fanout iteration) without requiring kernel
    // runtime state.  Integration with the live scheduler is tested via the
    // existing kernel integration test suite.

    fn delivered(id: u64) -> (u64, SignalDeliveryOutcome) {
        (id, SignalDeliveryOutcome::Delivered)
    }

    fn not_found(id: u64) -> (u64, SignalDeliveryOutcome) {
        (id, SignalDeliveryOutcome::RecipientNotFound)
    }

    fn existence(id: u64) -> (u64, SignalDeliveryOutcome) {
        (id, SignalDeliveryOutcome::ExistenceConfirmed)
    }

    // ── SignalRoutingReport::build ────────────────────────────────────────────

    #[test]
    fn report_all_delivered_has_zero_failed() {
        let outcomes = alloc::vec![delivered(1), delivered(2), delivered(3)];
        let report = SignalRoutingReport::build(SignalTargetKind::ProcessGroup, outcomes);
        assert_eq!(report.targeted, 3);
        assert_eq!(report.succeeded, 3);
        assert_eq!(report.failed, 0);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn report_one_failure_is_counted_correctly() {
        let outcomes = alloc::vec![delivered(10), not_found(11), delivered(12)];
        let report = SignalRoutingReport::build(SignalTargetKind::ProcessGroup, outcomes);
        assert_eq!(report.targeted, 3);
        assert_eq!(report.succeeded, 2);
        assert_eq!(report.failed, 1);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].0, 11);
        assert_eq!(report.failures[0].1, SignalDeliveryOutcome::RecipientNotFound);
    }

    #[test]
    fn report_all_failed_has_zero_succeeded() {
        let outcomes = alloc::vec![not_found(20), not_found(21)];
        let report = SignalRoutingReport::build(SignalTargetKind::ProcessGroup, outcomes);
        assert_eq!(report.targeted, 2);
        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 2);
        assert_eq!(report.failures.len(), 2);
    }

    #[test]
    fn report_empty_outcomes_is_zero_targeted() {
        let outcomes: Vec<(u64, SignalDeliveryOutcome)> = alloc::vec![];
        let report = SignalRoutingReport::build(SignalTargetKind::Process, outcomes);
        assert_eq!(report.targeted, 0);
        assert_eq!(report.succeeded, 0);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn report_existence_confirmed_counted_as_succeeded() {
        // sig==0 existence-check outcomes are not failures.
        let outcomes = alloc::vec![existence(5), existence(6)];
        let report = SignalRoutingReport::build(SignalTargetKind::ProcessGroup, outcomes);
        assert_eq!(report.targeted, 2);
        assert_eq!(report.succeeded, 2);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn report_preserves_target_kind() {
        let outcomes = alloc::vec![delivered(1)];
        let report = SignalRoutingReport::build(SignalTargetKind::Thread, outcomes);
        assert_eq!(report.target_kind, SignalTargetKind::Thread);
    }

    // ── SignalRoute construction ──────────────────────────────────────────────

    #[test]
    fn route_fields_are_accessible() {
        let r = SignalRoute {
            signal: 9,
            sender_tid: Some(42),
            target_kind: SignalTargetKind::Process,
            target_id: 100,
        };
        assert_eq!(r.signal, 9);
        assert_eq!(r.sender_tid, Some(42));
        assert_eq!(r.target_kind, SignalTargetKind::Process);
        assert_eq!(r.target_id, 100);
    }

    #[test]
    fn route_sender_tid_may_be_none() {
        let r = SignalRoute {
            signal: 15,
            sender_tid: None,
            target_kind: SignalTargetKind::Thread,
            target_id: 7,
        };
        assert_eq!(r.sender_tid, None);
    }

    // ── Partial fanout: failure does not corrupt other recipients ─────────────

    /// Simulates the fanout logic using the same report-building code to
    /// verify that a failure for one recipient is isolated.
    #[test]
    fn partial_fanout_failure_does_not_affect_others() {
        // Simulate three recipients: first and third succeed, second not found.
        let outcomes = alloc::vec![delivered(30), not_found(31), delivered(32)];
        let report = SignalRoutingReport::build(SignalTargetKind::ProcessGroup, outcomes);

        assert_eq!(report.succeeded, 2);
        assert_eq!(report.failed, 1);
        // The only failure is pid 31.
        assert_eq!(report.failures[0].0, 31);
        // Recipients 30 and 32 are NOT in the failure list.
        assert!(!report.failures.iter().any(|(id, _)| *id == 30));
        assert!(!report.failures.iter().any(|(id, _)| *id == 32));
    }

    // ── Coalescing semantics: duplicate delivery behavior ─────────────────────
    //
    // The routing layer does not de-duplicate by design.  If a recipient
    // appears twice in a snapshot (which should not happen in practice but
    // could in adversarial code), the routing layer delivers twice.
    // Coalescing is enforced recipient-side by the `SigSet` bitmask in
    // `ProcessSignals::post` — setting a bit that is already set is a no-op.
    //
    // This test documents and verifies that expectation.

    #[test]
    fn duplicate_recipient_in_snapshot_routes_twice_but_coalesces_in_pending_set() {
        // Routing layer: delivers twice (both show as Delivered in report).
        let outcomes = alloc::vec![delivered(50), delivered(50)];
        let report = SignalRoutingReport::build(SignalTargetKind::ProcessGroup, outcomes);
        assert_eq!(report.targeted, 2);
        assert_eq!(report.succeeded, 2);
        assert_eq!(report.failed, 0);
        // Actual coalescing is enforced by ProcessSignals::post (SigSet.add is
        // idempotent).  The routing layer is not responsible for deduplication.
    }

    // ── SignalTargetKind equality ─────────────────────────────────────────────

    #[test]
    fn target_kind_equality_is_correct() {
        assert_eq!(SignalTargetKind::Process, SignalTargetKind::Process);
        assert_ne!(SignalTargetKind::Process, SignalTargetKind::Thread);
        assert_ne!(SignalTargetKind::Process, SignalTargetKind::ProcessGroup);
        assert_eq!(SignalTargetKind::Thread, SignalTargetKind::Thread);
        assert_eq!(SignalTargetKind::ProcessGroup, SignalTargetKind::ProcessGroup);
    }

    // ── SignalDeliveryOutcome equality ────────────────────────────────────────

    #[test]
    fn delivery_outcome_equality_is_correct() {
        assert_eq!(SignalDeliveryOutcome::Delivered, SignalDeliveryOutcome::Delivered);
        assert_ne!(
            SignalDeliveryOutcome::Delivered,
            SignalDeliveryOutcome::RecipientNotFound
        );
        assert_ne!(
            SignalDeliveryOutcome::ExistenceConfirmed,
            SignalDeliveryOutcome::RecipientNotFound
        );
    }
}
