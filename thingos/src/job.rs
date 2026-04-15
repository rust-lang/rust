//! Canonical public types for the `thingos.job` schema kind.
//!
//! # Schema
//!
//! ## v1 (Phase 2) — lifecycle shape
//!
//! ```text
//! kind thingos.job.state = enum {
//!   New,
//!   Running,
//!   Exited,
//! }
//!
//! kind thingos.job = struct {
//!   state: thingos.job.state,
//! }
//! ```
//!
//! ## v2 (Phase 3) — exit and wait shapes
//!
//! ```text
//! kind thingos.job.exit = struct {
//!   state: thingos.job.state,
//!   code:  option<i32>,
//! }
//!
//! kind thingos.job.wait.result = enum {
//!   Running,
//!   Exited { code: option<i32> },
//! }
//! ```
//!
//! # Transitional mapping
//!
//! The current kernel `Process` structure is the *provisional* internal
//! backing for a canonical `Job`.  A dedicated bridge layer
//! (`kernel::job::bridge`) converts `Process` lifecycle state into this public
//! representation so that the new ontology appears first at the edges while
//! internal machinery is replaced gradually.
//!
//! The `Process` struct is deliberately *not* renamed or split here.  Future
//! phases will hollow it out by migrating individual responsibilities:
//!
//! * **Phase 3** (this phase): wait/exit reporting migrates into `Job` terms.
//! * **Phase 4**: introduce `Group` as the coordination truth.
//! * **Phase 5**: begin internal `Process` decomposition by responsibility.
//!
//! | Canonical field         | Current kernel source                                      |
//! |-------------------------|------------------------------------------------------------|
//! | `Job::state`            | inferred from the live `ThreadState` of the thread group   |
//! | `JobExit::state`        | `ThreadState::Dead` → `JobState::Exited`                   |
//! | `JobExit::code`         | `Thread::exit_code` (set by `mark_task_exited`)            |
//! | `JobWaitResult::Exited` | result of `poll_task_exit` / `waitpid` returning Some(code)|
//! | `JobWaitResult::Running`| result of `poll_task_exit` returning `None`                |
//!
//! # Note on `Process` vs `Job`
//!
//! Not all of `Process` will eventually become `Job`.  `Process` currently
//! carries address-space ownership, FD tables, namespaces, signal state, and
//! more.  Those responsibilities will migrate to their own canonical kinds
//! over time.  `Job` covers *only* the lifecycle/accounting axis: creation,
//! running, and exit.

/// The KindId generated for `thingos.job.state` by `kindc`.
///
/// Identifies the canonical job lifecycle-state enum kind.
pub const KIND_ID_THINGOS_JOB_STATE: [u8; 16] = [
    0x13, 0x8d, 0xf8, 0x73, 0x03, 0xbc, 0x87, 0xb8,
    0xdf, 0x0a, 0x02, 0x78, 0x55, 0x3a, 0x92, 0x50,
];

/// The KindId generated for `thingos.job` by `kindc`.
///
/// Identifies the canonical job lifecycle/accounting schema kind.  Consumers
/// that need to distinguish a `Job`-shaped message payload can compare against
/// this constant.
pub const KIND_ID_THINGOS_JOB: [u8; 16] = [
    0x31, 0xb2, 0x29, 0x72, 0x08, 0xbd, 0x27, 0x39,
    0xde, 0x66, 0x1c, 0x69, 0x2d, 0x4c, 0x0f, 0x99,
];

/// The KindId generated for `thingos.job.exit` by `kindc`.
///
/// Identifies the canonical job-exit notification schema kind.  Consumers
/// dispatching on [`KindId`](crate::message::KindId) should compare against
/// `KindId::THINGOS_JOB_EXIT` (derived from this value) to recognise job-exit
/// messages.
pub const KIND_ID_THINGOS_JOB_EXIT: [u8; 16] = [
    0xc2, 0x60, 0x8e, 0x30, 0xff, 0xa2, 0xa2, 0xda,
    0x8b, 0x96, 0x22, 0x8d, 0x3e, 0xd0, 0x11, 0x74,
];

/// The KindId generated for `thingos.job.wait.result` by `kindc`.
///
/// Identifies the canonical job wait-result schema kind.
pub const KIND_ID_THINGOS_JOB_WAIT_RESULT: [u8; 16] = [
    0x60, 0x9d, 0x28, 0x53, 0xea, 0xaf, 0x37, 0x97,
    0x2d, 0xef, 0x65, 0x61, 0x84, 0xbe, 0xd2, 0xe9,
];

/// Canonical lifecycle state for a `thingos.job`.
///
/// This is the external truth for process lifecycle exposed at system
/// boundaries.  The kernel's internal `Process`+`Thread` model maps into this
/// via `kernel::job::bridge`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobState {
    /// The job has been created but no threads are yet running.
    New,
    /// At least one thread in the job is alive (Runnable, Running, or Blocked).
    Running,
    /// All threads in the job have exited.
    Exited,
}

impl JobState {
    /// Return a short human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::New => "New",
            JobState::Running => "Running",
            JobState::Exited => "Exited",
        }
    }
}

/// Canonical public representation of a lifecycle/accounting container.
///
/// Corresponds to the `thingos.job` schema kind (v1).  The kernel's internal
/// `Process` structure is the current transitional backing; the `bridge` module
/// in `kernel::job` converts `Process`-shaped state into this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Job {
    /// The current lifecycle state of this job.
    pub state: JobState,
}

// ── Phase 3: exit and wait types ─────────────────────────────────────────────

/// Canonical description of an exited (or still-live) job.
///
/// Corresponds to the `thingos.job.exit` schema kind (v1).
///
/// When `state` is `JobState::Exited`, `code` holds the integer exit code
/// reported by the process.  For live jobs (`New` or `Running`) `code` is
/// `None`.
///
/// # Transitional mapping
///
/// The current kernel sets `Thread::exit_code` inside `mark_task_exited`.
/// `kernel::job::bridge::job_exit_from_snapshot` reads that field (via
/// `ProcessSnapshot`) to populate this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobExit {
    /// Lifecycle state at the time of the snapshot.
    pub state: JobState,
    /// Exit code, present only when `state == JobState::Exited`.
    pub code: Option<i32>,
}

impl JobExit {
    /// Format as a human-readable text blob suitable for procfs.
    ///
    /// Outputs two lines:
    /// ```text
    /// state: Running
    /// code: -
    /// ```
    /// or
    /// ```text
    /// state: Exited
    /// code: 0
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        alloc::format!(
            "state: {}\ncode: {}\n",
            self.state.as_str(),
            match self.code {
                Some(c) => alloc::format!("{}", c),
                None => alloc::string::String::from("-"),
            }
        )
    }

    /// Encode this `JobExit` plus `job_id` as a binary notification payload.
    ///
    /// The layout is 10 bytes:
    ///
    /// | Bytes | Type        | Value                                   |
    /// |-------|-------------|------------------------------------------|
    /// | 0–3   | u32 LE      | `job_id` (the PID of the exiting job)   |
    /// | 4     | u8          | state discriminant (0=New, 1=Running, 2=Exited) |
    /// | 5     | u8          | code_present (0 or 1)                   |
    /// | 6–9   | i32 LE      | exit code (only valid when `code_present == 1`) |
    ///
    /// Pair with [`decode_notification`] for round-trip testing.
    pub fn encode_as_notification(&self, job_id: u32) -> alloc::vec::Vec<u8> {
        let state_byte: u8 = match self.state {
            JobState::New => 0,
            JobState::Running => 1,
            JobState::Exited => 2,
        };
        let (code_present, code_bytes): (u8, [u8; 4]) = match self.code {
            Some(c) => (1, c.to_le_bytes()),
            None => (0, [0u8; 4]),
        };
        let mut buf = alloc::vec::Vec::with_capacity(10);
        buf.extend_from_slice(&job_id.to_le_bytes());
        buf.push(state_byte);
        buf.push(code_present);
        buf.extend_from_slice(&code_bytes);
        buf
    }

    /// Decode a binary notification payload produced by [`encode_as_notification`].
    ///
    /// Returns `Some((job_id, JobExit))` on success, or `None` if the slice is
    /// too short or contains an unrecognised state discriminant.
    pub fn decode_notification(bytes: &[u8]) -> Option<(u32, Self)> {
        if bytes.len() < 10 {
            return None;
        }
        let job_id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let state = match bytes[4] {
            0 => JobState::New,
            1 => JobState::Running,
            2 => JobState::Exited,
            _ => return None,
        };
        let code = if bytes[5] == 1 {
            Some(i32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]))
        } else {
            None
        };
        Some((job_id, JobExit { state, code }))
    }
}

/// The canonical result of waiting on (or polling) a job.
///
/// Corresponds to the `thingos.job.wait.result` schema kind (v1).
///
/// # Transitional mapping
///
/// * `poll_task_exit` returning `None`       → `JobWaitResult::Running`
/// * `poll_task_exit` returning `Some(code)` → `JobWaitResult::Exited { code: Some(code) }`
/// * `waitpid` returning `(pid, code)`       → `JobWaitResult::Exited { code: Some(code) }`
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobWaitResult {
    /// The job is still running; no exit code available yet.
    Running,
    /// The job has exited.  `code` is the integer exit code (always `Some`
    /// when produced by the current kernel, which always records a code).
    Exited {
        /// The exit code reported by the process.
        code: Option<i32>,
    },
}

impl JobWaitResult {
    /// Return a short label for the state part of the result.
    pub fn state_str(&self) -> &'static str {
        match self {
            JobWaitResult::Running => "Running",
            JobWaitResult::Exited { .. } => "Exited",
        }
    }

    /// Format as a human-readable text blob suitable for procfs.
    pub fn as_text(&self) -> alloc::string::String {
        match self {
            JobWaitResult::Running => alloc::string::String::from("result: Running\ncode: -\n"),
            JobWaitResult::Exited { code } => alloc::format!(
                "result: Exited\ncode: {}\n",
                match code {
                    Some(v) => alloc::format!("{}", v),
                    None => alloc::string::String::from("-"),
                }
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── JobState ─────────────────────────────────────────────────────────────

    #[test]
    fn test_job_state_as_str() {
        assert_eq!(JobState::New.as_str(), "New");
        assert_eq!(JobState::Running.as_str(), "Running");
        assert_eq!(JobState::Exited.as_str(), "Exited");
    }

    #[test]
    fn test_job_state_equality() {
        assert_eq!(JobState::New, JobState::New);
        assert_ne!(JobState::New, JobState::Running);
        assert_ne!(JobState::Running, JobState::Exited);
    }

    // ── Job ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_job_carries_state() {
        let job = Job { state: JobState::Running };
        assert_eq!(job.state, JobState::Running);
    }

    // ── JobExit ──────────────────────────────────────────────────────────────

    #[test]
    fn test_job_exit_as_text_running() {
        let exit = JobExit { state: JobState::Running, code: None };
        let text = exit.as_text();
        assert!(text.contains("state: Running"), "unexpected: {}", text);
        assert!(text.contains("code: -"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_exit_as_text_exited_with_code() {
        let exit = JobExit { state: JobState::Exited, code: Some(0) };
        let text = exit.as_text();
        assert!(text.contains("state: Exited"), "unexpected: {}", text);
        assert!(text.contains("code: 0"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_exit_as_text_exited_nonzero() {
        let exit = JobExit { state: JobState::Exited, code: Some(42) };
        let text = exit.as_text();
        assert!(text.contains("code: 42"), "unexpected: {}", text);
    }

    // ── JobExit::encode_as_notification / decode_notification ────────────────

    #[test]
    fn encode_notification_produces_ten_bytes() {
        let exit = JobExit { state: JobState::Exited, code: Some(0) };
        let bytes = exit.encode_as_notification(1234);
        assert_eq!(bytes.len(), 10);
    }

    #[test]
    fn encode_decode_round_trip_exited_with_code() {
        let original = JobExit { state: JobState::Exited, code: Some(42) };
        let job_id = 99u32;
        let bytes = original.encode_as_notification(job_id);
        let (decoded_id, decoded_exit) = JobExit::decode_notification(&bytes).expect("should decode");
        assert_eq!(decoded_id, job_id);
        assert_eq!(decoded_exit, original);
    }

    #[test]
    fn encode_decode_round_trip_exited_no_code() {
        let original = JobExit { state: JobState::Exited, code: None };
        let bytes = original.encode_as_notification(7);
        let (decoded_id, decoded_exit) = JobExit::decode_notification(&bytes).expect("should decode");
        assert_eq!(decoded_id, 7);
        assert_eq!(decoded_exit, original);
    }

    #[test]
    fn encode_decode_round_trip_running() {
        let original = JobExit { state: JobState::Running, code: None };
        let bytes = original.encode_as_notification(1);
        let (decoded_id, decoded) = JobExit::decode_notification(&bytes).expect("should decode");
        assert_eq!(decoded_id, 1);
        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_new() {
        let original = JobExit { state: JobState::New, code: None };
        let bytes = original.encode_as_notification(0);
        let (decoded_id, decoded) = JobExit::decode_notification(&bytes).expect("should decode");
        assert_eq!(decoded_id, 0);
        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_negative_exit_code() {
        let original = JobExit { state: JobState::Exited, code: Some(-1) };
        let bytes = original.encode_as_notification(500);
        let (decoded_id, decoded) = JobExit::decode_notification(&bytes).expect("should decode");
        assert_eq!(decoded_id, 500);
        assert_eq!(decoded.code, Some(-1));
    }

    #[test]
    fn decode_notification_too_short_returns_none() {
        assert!(JobExit::decode_notification(&[]).is_none());
        assert!(JobExit::decode_notification(&[0u8; 9]).is_none());
    }

    #[test]
    fn decode_notification_bad_state_byte_returns_none() {
        let mut bytes = [0u8; 10];
        bytes[4] = 0xff; // invalid state discriminant
        assert!(JobExit::decode_notification(&bytes).is_none());
    }

    #[test]
    fn encode_decode_large_job_id() {
        let original = JobExit { state: JobState::Exited, code: Some(127) };
        let job_id = u32::MAX;
        let bytes = original.encode_as_notification(job_id);
        let (decoded_id, decoded) = JobExit::decode_notification(&bytes).expect("should decode");
        assert_eq!(decoded_id, job_id);
        assert_eq!(decoded, original);
    }

    // ── JobWaitResult ────────────────────────────────────────────────────────

    #[test]
    fn test_job_wait_result_state_str() {
        assert_eq!(JobWaitResult::Running.state_str(), "Running");
        assert_eq!(JobWaitResult::Exited { code: Some(0) }.state_str(), "Exited");
        assert_eq!(JobWaitResult::Exited { code: None }.state_str(), "Exited");
    }

    #[test]
    fn test_job_wait_result_as_text_running() {
        let text = JobWaitResult::Running.as_text();
        assert!(text.contains("result: Running"), "unexpected: {}", text);
        assert!(text.contains("code: -"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_wait_result_as_text_exited_with_code() {
        let text = JobWaitResult::Exited { code: Some(1) }.as_text();
        assert!(text.contains("result: Exited"), "unexpected: {}", text);
        assert!(text.contains("code: 1"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_wait_result_as_text_exited_no_code() {
        let text = JobWaitResult::Exited { code: None }.as_text();
        assert!(text.contains("result: Exited"), "unexpected: {}", text);
        assert!(text.contains("code: -"), "unexpected: {}", text);
    }

    #[test]
    fn test_job_wait_result_equality() {
        assert_eq!(JobWaitResult::Running, JobWaitResult::Running);
        assert_ne!(JobWaitResult::Running, JobWaitResult::Exited { code: Some(0) });
        assert_eq!(
            JobWaitResult::Exited { code: Some(0) },
            JobWaitResult::Exited { code: Some(0) }
        );
        assert_ne!(
            JobWaitResult::Exited { code: Some(0) },
            JobWaitResult::Exited { code: Some(1) }
        );
    }
}
