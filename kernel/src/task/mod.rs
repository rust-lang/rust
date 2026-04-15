pub mod bridge;
pub mod exec;
pub mod loader;
pub mod registry;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;

use abi::types::StackInfo;
use spin::Mutex;

pub use crate::sched::Scheduler;
use crate::simd::SimdState;
use crate::{BootRuntime, BootTasking, sched as scheduler};

pub type ThreadId = crate::sched::state::ThreadId;
/// Backward-compatible alias — prefer `ThreadId` in new code.
pub type TaskId = ThreadId;
pub use crate::sched::state::{Affinity, ThreadPriority, ThreadState};
/// Backward-compatible alias — prefer `ThreadPriority` in new code.
pub type TaskPriority = ThreadPriority;
/// Backward-compatible alias — prefer `ThreadState` in new code.
pub type TaskState = ThreadState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupArg {
    None,
    BootRegistry,
    DeviceId(u64),
    Raw(usize),
}

impl StartupArg {
    pub fn to_raw(self) -> usize {
        match self {
            StartupArg::None => 0,
            StartupArg::BootRegistry => 0x600000,
            StartupArg::DeviceId(id) => id as usize,
            StartupArg::Raw(val) => val,
        }
    }
}

/// Address-space subdivision of a `Process`.
///
/// This struct groups all memory/address-space concerns that are conceptually
/// owned by **Space**, not by `Process`, `Job`, or `Task`.  It lives inside
/// `Process` as a transitional measure: the fields are here because a
/// first-class `Space` kernel object does not yet exist, but they are
/// deliberately separated so that future extraction into `Space` is obvious
/// and mechanical.
///
/// # Conceptual future ownership
///
/// In the emerging ThingOS object model:
/// - **Task** owns execution context (registers, stack, scheduling state).
/// - **Job** owns process lifecycle (creation, exit, reaping).
/// - **Space** owns address-space state: the page-table root, the virtual
///   memory map, user-stack / program-image layout, and all mapping metadata.
///
/// New kernel code that needs to add memory-ownership state should add it
/// **here**, not directly to `Process`.  When `Space` is introduced as a
/// first-class object this subdivision is the extraction seam.
///
/// # Future extraction seams
///
/// Likely follow-on cuts once a first-class `Space` is introduced:
/// - Promote `ProcessAddressSpace` into `Space` and share it across processes
///   that map the same image (copy-on-write / shared mappings).
/// - Separate the `exec` transition helpers that currently live in
///   `kernel::task::exec` and reach into this struct directly.
/// - Expose canonical `Space` identity at syscall boundaries
///   (e.g. `SYS_VM_MAP` currently operates on the implicit current space).
/// - Untangle `fork`/`exec` assumptions: today `exec` replaces both
///   `mappings` and `aspace_raw` in-place; a real `Space` swap would instead
///   atomically replace the entire `ProcessAddressSpace`.
pub struct ProcessAddressSpace {
    /// Process-scoped VM mapping list.
    ///
    /// Every `Thread` in this process holds a clone of this `Arc` in its own
    /// `mappings` field so the scheduler's hot path (the `CURRENT_MAPPINGS`
    /// per-CPU cache) works without locking the `Process` mutex on every
    /// context switch.  The underlying `MappingList` is therefore always the
    /// same object visible from both `Process.space.mappings` and each
    /// thread's `Thread.mappings`.
    pub mappings: alloc::sync::Arc<spin::Mutex<crate::memory::mappings::MappingList>>,

    /// Process-scoped address-space token (architecture-specific raw value).
    ///
    /// Stores the page-table root for this process in an architecture-neutral
    /// `u64` representation (see [`crate::BootTasking::aspace_to_raw`]).  All
    /// threads in this process share the same address space; `Thread.aspace`
    /// holds a typed copy of the same token for the scheduler's fast path.
    ///
    /// Updated atomically with `Thread.aspace` during exec and remains 0
    /// for kernel-only threads (which have no user address space).
    pub aspace_raw: u64,

    /// First-class `Space` kernel object (Phase 1 — transitional).
    ///
    /// Wraps the same `mappings` `Arc` and carries a stable
    /// [`crate::space::Space::id`] (`SpaceId`) for diagnostics and future
    /// handle-based ABI exposure.
    ///
    /// In Phase 1 this lives inside `ProcessAddressSpace` (alongside
    /// `mappings` and `aspace_raw`) so that **no call site that constructs
    /// `ProcessAddressSpace` needs to change** — the constructors
    /// (`empty`, `from_parts`) automatically create the corresponding
    /// `Space` object.  Once the extraction is complete, this field will
    /// become the sole VM anchor and `mappings`/`aspace_raw` will move
    /// inside `Space`.
    ///
    /// # Usage rules
    ///
    /// * Read `space.space_obj.id` when you need a stable `SpaceId`.
    /// * Read `space.space_obj.mapping_count()` for a diagnostic region count.
    /// * Continue to use `space.mappings` and `space.aspace_raw` for all
    ///   existing VM-mutation paths — they share the same `Arc` allocation.
    pub space_obj: alloc::sync::Arc<crate::space::Space>,
}

impl ProcessAddressSpace {
    /// Create a fresh, empty address space subdivision (no mappings, no page-table root).
    ///
    /// Used when spawning a new process before ELF loading assigns a real
    /// address space.  A fresh `Space` kernel object is created automatically
    /// and shares the same `Arc<Mutex<MappingList>>`.
    pub fn empty() -> Self {
        let mappings = alloc::sync::Arc::new(spin::Mutex::new(
            crate::memory::mappings::MappingList::new(),
        ));
        let space_obj = alloc::sync::Arc::new(crate::space::Space {
            id: crate::space::alloc_space_id(),
            mappings: alloc::sync::Arc::clone(&mappings),
            aspace_raw: 0,
        });
        ProcessAddressSpace { mappings, aspace_raw: 0, space_obj }
    }

    /// Create an address space subdivision from an existing mappings `Arc` and
    /// a raw page-table token.
    ///
    /// Used by the spawn path after the ELF loader has populated the address
    /// space and the architecture runtime has converted the handle to a raw token.
    /// A fresh `Space` kernel object is created automatically and shares the
    /// provided `mappings` `Arc`.
    pub fn from_parts(
        mappings: alloc::sync::Arc<spin::Mutex<crate::memory::mappings::MappingList>>,
        aspace_raw: u64,
    ) -> Self {
        let space_obj = alloc::sync::Arc::new(crate::space::Space {
            id: crate::space::alloc_space_id(),
            mappings: alloc::sync::Arc::clone(&mappings),
            aspace_raw,
        });
        ProcessAddressSpace { mappings, aspace_raw, space_obj }
    }
}

/// Lifecycle subdivision of a `Process`.
///
/// This struct groups all lifecycle/accounting concerns that are conceptually
/// owned by **Job**, not by `Process`, `Task`, or `Space`.  It lives inside
/// `Process` as a transitional measure: the fields are here because a
/// first-class `Job` kernel object does not yet exist, but they are
/// deliberately separated so that future extraction into `Job` is obvious
/// and mechanical.
///
/// # Conceptual future ownership
///
/// In the emerging ThingOS object model:
/// - **Task** owns execution context (registers, stack, scheduling state).
/// - **Job** owns lifecycle: creation, parent/child linkage, thread-group
///   membership, exec gating, exit-status accumulation, and reaping.
/// - **Process** is transitional and should stop accumulating lifecycle meaning.
///
/// New kernel code that needs to add lifecycle state should add it **here**,
/// not directly to top-level `Process`.  When `Job` is introduced as a
/// first-class object, this subdivision is the extraction seam.
///
/// # Current lifecycle responsibility set
///
/// | Field              | Role                                                        |
/// |--------------------|-------------------------------------------------------------|
/// | `ppid`             | Parent/child linkage for `waitpid` filtering                |
/// | `thread_ids`       | Thread-group membership; drives group-exit and exec collapse|
/// | `exec_in_progress` | Lifecycle gate; blocks `SYS_SPAWN_THREAD` during exec       |
/// | `children_done`    | Exit-status accumulator consumed by parent `waitpid`        |
///
/// # Future extraction seams
///
/// Likely follow-on cuts once a first-class `Job` is introduced:
/// - Promote `ProcessLifecycle` into `Job` and share it across threads in the
///   group (replacing the `Arc<Mutex<Process>>` back-reference for lifecycle).
/// - Move `exit_code` and `exit_waiters` from `Thread<R>` here, then into `Job`.
/// - Move parent/child lifecycle semantics (SIGCHLD, orphan reaping) into `Job`.
/// - Move `pid` (TGID) from top-level `Process` here once `Space` identity is
///   separated; `pid` currently doubles as both lifecycle ID (→ `Job`) and
///   address-space tag (→ `Space`).
/// - Detach `children_done` from `Process` entirely once `Job` can hold its own
///   wait queue.
///
/// # Relationship to the `kernel::job::bridge` module
///
/// `kernel::job::bridge` is the canonical public surface for lifecycle state.
/// All lifecycle-facing public paths (procfs, introspection, wait syscalls)
/// should derive `Job` / `JobExit` / `JobWaitResult` through that bridge
/// rather than reading this subdivision directly.  This subdivision is the
/// preferred **source** for those bridge mappings.
pub struct ProcessLifecycle {
    /// PID of the parent process.
    ///
    /// Used by `waitpid` to filter exit notifications to the correct parent.
    /// Migrates to `Job` with exit/wait semantics.
    pub ppid: u32,
    /// TIDs of all threads in this thread group.
    ///
    /// The first entry is the thread-group leader (TID == PID).  Entries are
    /// added on `spawn_user_thread` and removed when a thread exits.
    /// Drives group-leader exit (kills siblings) and exec collapse.
    pub thread_ids: Vec<ThreadId>,
    /// Set while an `exec` is in progress; blocks new `SYS_SPAWN_THREAD` calls.
    ///
    /// This is a lifecycle gate: concurrent thread spawning is illegal during
    /// exec collapse.  Belongs with `Job` lifecycle once extracted.
    pub exec_in_progress: bool,
    /// Exited children waiting for `waitpid` to consume their status.
    ///
    /// Each entry is `(child_pid, wait_status)`.  The status is encoded in
    /// the same format as POSIX `waitpid`: normal exit uses `(code << 8)`,
    /// signal termination uses `signum`, and stopped/continued children use
    /// the appropriate `w_stop_sig` / `w_continued` values.
    pub children_done: alloc::collections::VecDeque<(u32, i32)>,
    /// Optional inbox to receive a canonical `JobExit` message when this job
    /// exits.
    ///
    /// When set, `kernel::job::notify::emit_job_exit` delivers a typed
    /// `Message` (kind `THINGOS_JOB_EXIT`) to this inbox at the moment the
    /// thread-group leader transitions to the `Dead` state.
    ///
    /// Set via `kernel::job::notify::register_exit_observer`.  `None` means
    /// no inbox delivery is attempted (legacy/transitional path only).
    pub exit_observer_inbox: Option<crate::inbox::InboxId>,
}

impl ProcessLifecycle {
    /// Create a fresh lifecycle subdivision for a new process.
    ///
    /// `ppid` is the PID of the creating (parent) process, or `0` for a
    /// root/orphan process.  `leader_tid` is the TID of the thread-group
    /// leader (normally equal to the new process's PID cast to `ThreadId`).
    pub fn new(ppid: u32, leader_tid: ThreadId) -> Self {
        ProcessLifecycle {
            ppid,
            thread_ids: alloc::vec![leader_tid],
            exec_in_progress: false,
            children_done: alloc::collections::VecDeque::new(),
            exit_observer_inbox: None,
        }
    }
}

/// Delivery strategy used when a message is enqueued into a process inbox.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageDeliveryKind {
    /// One sender targeted one specific recipient.
    Direct,
    /// One sender targeted a group and this recipient was reached via fanout.
    GroupBroadcast,
}

/// Delivery metadata carried alongside a queued process message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessMessageMetadata {
    /// Thread ID of the sender at enqueue time.
    pub sender_tid: ThreadId,
    /// Provisional sender Job ID (currently the sender process PID), if known.
    pub sender_job: Option<u32>,
    /// Target process-group ID for group fanout deliveries.
    pub target_group: Option<u32>,
    /// Delivery strategy used to enqueue the message.
    pub delivery_kind: MessageDeliveryKind,
    /// Per-broadcast sequence index assigned during fanout.
    pub broadcast_sequence: Option<u64>,
}

/// Message record stored in a process inbox.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessMessage {
    /// Canonical typed message envelope.
    pub message: crate::message::Message,
    /// Metadata about how this message was delivered.
    pub metadata: ProcessMessageMetadata,
}

/// Process inbox enqueue errors for prototype typed delivery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageEnqueueError {
    /// Inbox reached its bounded capacity.
    InboxFull { capacity: usize },
}

/// Fixed prototype inbox capacity per process.
pub const PROCESS_MESSAGE_INBOX_CAPACITY: usize = 64;

/// Unix legacy compatibility state carried by a `Process`.
///
/// This struct is the **explicit quarantine boundary** for all Unix-derived
/// state that is kept for compatibility but is **not** architectural truth in
/// ThingOS.  It lives inside [`Process`] as a named subdivision so that:
///
/// * A reader immediately sees which concepts are transitional legacy.
/// * New code is forced to touch `unix_compat.*` rather than top-level
///   `Process` fields, making every new use of legacy state visible at code
///   review time.
/// * The extraction seam for each field is documented here, not scattered.
///
/// # ⚠️ Do not add new fields here
///
/// New kernel code MUST NOT add fields to `ProcessUnixCompat` unless
/// absolutely forced by a Unix compatibility requirement.  All new public
/// paths should use Task / Job / Group / Place / Authority vocabulary instead.
///
/// # Field inventory and future homes
///
/// | Field            | Unix origin                    | Future ThingOS home              |
/// |------------------|--------------------------------|----------------------------------|
/// | `signals`        | POSIX per-process signal state | Message / Inbox / Group broadcast|
/// | `pgid`           | Process-group ID               | Group (Phase 5)                  |
/// | `sid`            | Session ID                     | Group / Presence / Place (Phase 5)|
/// | `session_leader` | Session-leader flag            | Group / Presence (Phase 5)       |
/// | `argv`           | Spawn-time argument vector     | Structured spawn record → Job    |
/// | `env`            | Inherited environment map      | Place / Authority context        |
/// | `auxv`           | ELF auxiliary vector           | Structured spawn record → Job    |
///
/// # Note on Presence
///
/// Presence (terminal attachment, UI/console ownership, person-in-place) has
/// not yet been introduced.  When it is, controlling-TTY state currently
/// embedded in `signals` (SIGTTOU/SIGTTIN, job-control stop) will move there,
/// **not** into Place.
pub struct ProcessUnixCompat {
    /// Per-process signal state: dispositions, pending set, stop/alarm state.
    ///
    /// Includes the controlling-terminal glue (SIGTTOU/SIGTTIN, job-control
    /// stop signals) that properly belongs to Group / Presence once those
    /// concepts are introduced.
    ///
    /// FUTURE: → Message / Inbox / Group broadcast
    pub signals: crate::signal::ProcessSignals,

    /// Prototype typed message inbox used by direct and group fanout delivery.
    ///
    /// This queue is deliberately bounded to force explicit partial-failure
    /// behavior in early broadcast prototypes.
    pub message_inbox: VecDeque<ProcessMessage>,

    /// Process group ID.
    ///
    /// FUTURE: → Group (Phase 5)
    pub pgid: u32,

    /// Session ID.
    ///
    /// FUTURE: → Group / Presence / Place (Phase 5)
    pub sid: u32,

    /// True when this process is the leader of its session.
    ///
    /// FUTURE: → Group / Presence (Phase 5)
    pub session_leader: bool,

    /// Argument vector passed at spawn/exec time.
    ///
    /// FUTURE: → structured spawn record attached to Job
    pub argv: Vec<Vec<u8>>,

    /// Environment variables inherited at spawn/exec time.
    ///
    /// FUTURE: → Place / Authority context propagation
    pub env: BTreeMap<Vec<u8>, Vec<u8>>,

    /// ELF auxiliary vector (AT_* entries as `(type, value)` pairs).
    ///
    /// FUTURE: → structured spawn record attached to Job
    pub auxv: Vec<(u64, u64)>,
}

impl ProcessUnixCompat {
    /// Create Unix compatibility state for an **isolated** process (no parent).
    ///
    /// Sets `pgid` and `sid` to `pid`.  Pass `is_session_leader = true` when
    /// this process bootstraps a new session (e.g. the root init process);
    /// `false` otherwise.
    pub fn isolated(pid: u32, is_session_leader: bool) -> Self {
        ProcessUnixCompat {
            signals: crate::signal::ProcessSignals::new(),
            message_inbox: VecDeque::new(),
            pgid: pid,
            sid: pid,
            session_leader: is_session_leader,
            argv: Vec::new(),
            env: BTreeMap::new(),
            auxv: Vec::new(),
        }
    }

    /// Create Unix compatibility state for a **child** process that inherits
    /// session/group membership from `parent`.
    ///
    /// The child is never a session leader, inherits `pgid`/`sid` from the
    /// parent, and inherits the parent's environment map.
    pub fn inherit(parent: &ProcessUnixCompat) -> Self {
        ProcessUnixCompat {
            signals: crate::signal::ProcessSignals::new(),
            message_inbox: VecDeque::new(),
            pgid: parent.pgid,
            sid: parent.sid,
            session_leader: false,
            argv: Vec::new(),
            env: parent.env.clone(),
            auxv: Vec::new(),
        }
    }

    /// Enqueue one typed message into the process inbox.
    pub fn enqueue_message(&mut self, msg: ProcessMessage) -> Result<(), MessageEnqueueError> {
        if self.message_inbox.len() >= PROCESS_MESSAGE_INBOX_CAPACITY {
            return Err(MessageEnqueueError::InboxFull {
                capacity: PROCESS_MESSAGE_INBOX_CAPACITY,
            });
        }
        self.message_inbox.push_back(msg);
        Ok(())
    }

    /// Pop the next queued typed message from the process inbox.
    pub fn dequeue_message(&mut self) -> Option<ProcessMessage> {
        self.message_inbox.pop_front()
    }

    /// Current inbox depth for diagnostics and tests.
    pub fn message_inbox_len(&self) -> usize {
        self.message_inbox.len()
    }
}

/// First-class process object.
///
/// A `Process` is the unit of resource ownership in Thing-OS.  Every user
/// process has exactly one `Process`, shared via `Arc<Mutex<Process>>` by all
/// threads that belong to it.  Kernel-only threads have `process_info = None`.
///
/// # Ownership model
///
/// | Resource                | Owner   | How threads access it                            |
/// |-------------------------|---------|--------------------------------------------------|
/// | PID                     | Process | `process.lock().pid`                             |
/// | PPID / thread list      | Process | `process.lock().lifecycle.ppid` etc.             |
/// | VM address space        | Process | `process.lock().space.aspace_raw`                |
/// | VM mappings             | Process | `process.lock().space.mappings` (Arc)            |
/// | FD table                | Process | `process.lock().fd_table`                        |
/// | CWD                     | Process | `process.lock().cwd`                             |
/// | VFS namespace           | Process | `process.lock().namespace`                       |
/// | Unix compat (legacy)    | Process | `process.lock().unix_compat.*`                   |
/// | exec path               | Process | `process.lock().exec_path`                       |
///
/// # Locking rules
///
/// The `Process` mutex must **never** be acquired while the scheduler lock
/// (`SCHEDULER.lock()`) is held — reverse order causes deadlock.
///
/// # Responsibility classification (Phase 9 migration inventory)
///
/// Fields in this struct are grouped by their intended canonical destination
/// in the phased migration.  Fields are **not** yet extracted; they remain
/// here as transitional backing.  All public-facing access should go through
/// the appropriate bridge module.
///
/// **Lifecycle** (Job — `kernel::job::bridge`):
/// * `lifecycle` — grouped under [`ProcessLifecycle`]; contains `ppid`,
///   `thread_ids`, `exec_in_progress`, and `children_done`.
///   New code must not add lifecycle state directly to `Process`; add it to
///   `ProcessLifecycle` instead.
///
/// **Place** (world/visibility context — `kernel::place::bridge`):
/// * `cwd` — current working directory path → `Place::cwd`
/// * `namespace` — VFS mount-table view → `Place::namespace`
/// * *(no root field yet)* — effective filesystem root → `Place::root`
///
/// **Unix legacy compatibility** (quarantined — see [`ProcessUnixCompat`]):
/// * `unix_compat` — all Unix-derived state (`signals`, `pgid`, `sid`,
///   `session_leader`, `argv`, `env`, `auxv`).
///   These fields are NOT architectural truth; they are kept behind an
///   explicit compatibility boundary.  New code MUST NOT add to
///   `ProcessUnixCompat` without a Unix compatibility justification.
///
/// **Group** (coordination domain — `kernel::group::bridge`):
/// * `unix_compat.pgid` / `unix_compat.sid` / `unix_compat.session_leader`
///   — also used by Group bridge today
///
/// **Authority** (permission context — `kernel::authority::bridge`):
/// * `exec_path` — used as authority name fallback today
///
/// **Identity** (shared between Job and Space — not yet extracted):
/// * `pid` — TGID; doubles as lifecycle ID (→ `Job`) and address-space tag
///   (→ `Space`).  Split deferred until `Space` identity is separated.
///
/// **Space** (address-space ownership — future `Space` kernel object):
/// * `space` — grouped under [`ProcessAddressSpace`]; contains `mappings` and
///   `aspace_raw`.  New code must not attach additional memory-ownership state
///   directly to `Process`; add it to `ProcessAddressSpace` instead.
///
/// See `docs/concepts/process-object.md` for the full design document.
/// See `docs/migration/concept-mapping.md` §2 (Process → Job + Space + Authority + Place + Task(s))
/// for the canonical legacy→Janix mapping and naming rules.
pub struct Process {
    /// Thread Group ID — the PID of the thread-group leader.
    ///
    /// `pid` is kept at the top level of `Process` rather than inside
    /// `lifecycle` because it currently doubles as both the lifecycle identity
    /// (→ future `Job`) and the address-space identity (→ future `Space`).
    /// Once those two responsibilities are separated, `pid` will migrate into
    /// `ProcessLifecycle` alongside the other `Job`-bound fields.
    pub pid: u32,

    // ── Lifecycle subdivision (future `Job` kernel object) ────────────────────
    // Lifecycle concerns are grouped here rather than scattered across `Process`.
    // This subdivision is the extraction seam for a future first-class `Job`
    // object.  Do NOT add new lifecycle state directly to top-level `Process` —
    // add it to `ProcessLifecycle` instead.
    /// Lifecycle subdivision — conceptually future `Job` ownership.
    ///
    /// Contains parent/child linkage, thread-group membership, exec gating,
    /// and the `waitpid` exit queue.  See [`ProcessLifecycle`] for the full
    /// design rationale and future extraction seams.
    pub lifecycle: ProcessLifecycle,

    // ── Unix legacy compatibility boundary ───────────────────────────────────
    // ALL Unix-derived compatibility state lives here, behind an explicit
    // named boundary.  This makes legacy baggage visible in every code review.
    //
    // DO NOT move fields out of `unix_compat` back into top-level `Process`.
    // DO NOT add new Unix compatibility state to top-level `Process` directly.
    // Any unavoidable Unix compatibility code must live inside this boundary.
    /// Unix legacy compatibility state — signals, pgid, sid, argv, env, auxv.
    ///
    /// See [`ProcessUnixCompat`] for the full field inventory, future homes,
    /// and the rules that govern this boundary.
    pub unix_compat: ProcessUnixCompat,

    // ── Resource table ────────────────────────────────────────────────────────
    // Future: fd_table will move to a resource-authority domain.  For now it
    // remains as transitional Process baggage.
    /// File descriptor table — fds 0/1/2 pre-populated at spawn time.
    pub fd_table: crate::vfs::fd_table::FdTable,

    // ── Place context (Phase 8 — world/visibility boundary) ──────────────────
    // These fields answer "in what world does this execution happen?".
    // They feed `kernel::place::bridge` → `thingos::place::Place`.
    // They remain in `Process` as transitional backing; the canonical surface
    // is through the place bridge, not direct field access from new code.
    /// VFS namespace — currently a global stub shared by all processes.
    ///
    /// [`crate::vfs::NamespaceRef`] is a unit struct today: all instances
    /// resolve to the same underlying global mount table.  The field exists
    /// so that per-process namespace isolation can be added later without
    /// touching every spawn call site.
    ///
    /// Feeds `Place::namespace` through `kernel::place::bridge`.
    ///
    /// See `docs/concepts/namespaces.md` for the behaviour matrix and roadmap.
    pub namespace: crate::vfs::NamespaceRef,

    /// Current working directory.
    ///
    /// Feeds `Place::cwd` through `kernel::place::bridge`.
    /// New code must not read this field directly for world-context purposes;
    /// use `crate::place::bridge::place_from_snapshot` instead.
    pub cwd: alloc::string::String,

    /// Path of the currently-running executable image.
    pub exec_path: alloc::string::String,

    // ── Space context (future `Space` kernel object) ──────────────────────────
    // Address-space concerns are grouped here rather than scattered across
    // `Process`.  This subdivision is the extraction seam for a future
    // first-class `Space` object.  Do NOT attach new memory-ownership state
    // directly to `Process` — add it to `ProcessAddressSpace` instead.
    /// Address-space subdivision — conceptually future `Space` ownership.
    ///
    /// Contains the VM mapping list, the architecture-specific page-table
    /// token, and the first-class [`crate::space::Space`] object wrapper.
    /// See [`ProcessAddressSpace`] for the full design rationale.
    pub space: ProcessAddressSpace,
}

/// Backward-compatible alias — prefer `Process` in new code.
pub type ProcessInfo = Process;

/// Kernel representation of a single thread of execution.
///
/// Each `Thread<R>` corresponds to exactly one schedulable entity.  User
/// threads are created by `spawn_user_thread`; kernel threads by `spawn`.
///
/// # Fields split by concern
///
/// | Concern              | Fields                                               |
/// |----------------------|------------------------------------------------------|
/// | Identity             | `id`                                                 |
/// | Scheduler state      | `state`, `priority`, `timeslice_remaining`, …        |
/// | Execution context    | `ctx`, `kstack_*`, `aspace`, `user_fs_base`          |
/// | Per-thread flags     | `is_user`, `detached`, `pending_interrupt`           |
/// | Process reference    | `process_info` — shared `Arc<Mutex<Process>>`        |
/// | VM fast-path cache   | `mappings` — clone of `Process.mappings` (same Arc)  |
pub struct Thread<R: BootRuntime> {
    pub id: ThreadId,
    pub state: ThreadState,
    pub priority: ThreadPriority,
    pub exit_code: Option<i32>,
    /// Exit notification: level-triggered so status stays readable after wake.
    pub exit_waiters: crate::sched::WaitQueue,
    pub is_user: bool,
    pub wake_pending: bool,
    pub pending_interrupt: bool,
    pub affinity: Affinity,

    pub kstack_base: *mut u8,
    pub kstack_size: usize,
    pub kstack_top: u64,

    pub ctx: <R::Tasking as BootTasking>::Context,
    pub aspace: <R::Tasking as BootTasking>::AddressSpace,

    pub simd: SimdState,

    pub stack_info: Option<StackInfo>,

    /// VM mapping list — a clone of `Process.space.mappings` (same underlying `Arc`).
    ///
    /// Kept here for zero-lock fast access by the scheduler's per-CPU mapping
    /// cache (`CURRENT_MAPPINGS`).  Always updated atomically with
    /// `Process.space.mappings` during exec or thread creation.
    pub mappings: Arc<Mutex<crate::memory::mappings::MappingList>>,

    /// Remaining time slice in ticks before preemption.
    pub timeslice_remaining: u32,
    pub last_cpu: Option<usize>,

    /// Short human-readable name (e.g. "bristle", "idle/0").
    pub name: [u8; 32],
    pub name_len: u8,

    /// Owning process (shared across all threads in the group).
    ///
    /// `None` for pure kernel threads.
    pub process_info: Option<Arc<Mutex<Process>>>,

    /// Anti-starvation: tick when this thread was last enqueued.
    pub enqueued_at_tick: u64,

    /// Base priority before any aging boost.
    pub base_priority: ThreadPriority,

    /// Per-thread user-mode TLS base (FS_BASE on x86_64).
    pub user_fs_base: u64,

    /// `true` if this thread was created as detached (cannot be joined).
    pub detached: bool,

    /// Per-thread signal mask and thread-directed pending signals.
    pub signals: crate::signal::ThreadSignals,
}

/// Backward-compatible alias — prefer `Thread<R>` in new code.
pub type Task<R> = Thread<R>;

pub fn init<R: BootRuntime>() {
    crate::task::registry::init::<R>();
    crate::sched::init::<R>();
}
pub fn spawn<R: BootRuntime>(
    entry: extern "C" fn(usize) -> !,
    arg: StartupArg,
    priority: ThreadPriority,
    affinity: Affinity,
) -> ThreadId {
    crate::sched::spawn::<R>(entry, arg, priority, affinity)
}

pub fn spawn_with_priority<R: BootRuntime>(
    entry: extern "C" fn(usize) -> !,
    arg: StartupArg,
    priority: ThreadPriority,
) -> ThreadId {
    crate::sched::spawn_with_priority::<R>(entry, arg, priority)
}

pub unsafe fn block_current_erased() {
    unsafe {
        crate::sched::block_current_erased();
    }
}

pub unsafe fn wake_task_erased(tid: u64) {
    unsafe {
        crate::sched::wake_task_erased(tid);
    }
}

pub fn yield_now<R: BootRuntime>() -> bool {
    crate::sched::yield_now::<R>()
}

pub fn preempt_disable<R: BootRuntime>() {
    let rt = crate::runtime::<R>();
    let irq = rt.irq_disable();
    {
        let lock = crate::sched::SCHEDULER.lock();
        if let Some(ptr) = *lock {
            let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
            sched.preempt_disable();
        }
    }
    rt.irq_restore(irq);
}

pub fn preempt_enable<R: BootRuntime>() {
    let rt = crate::runtime::<R>();
    let irq = rt.irq_disable();

    let switch_params = {
        let lock = crate::sched::SCHEDULER.lock();
        if let Some(ptr) = *lock {
            let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
            sched.preempt_enable()
        } else {
            None
        }
    };

    if let Some(switch) = switch_params {
        let cr3_before = rt.debug_active_aspace_root();

        rt.tasking().activate_address_space(switch.to_aspace);

        let _cr3_after = rt.debug_active_aspace_root();

        unsafe {
            rt.tasking().switch_with_tls(
                &mut *switch.from_ctx,
                &*switch.to_ctx,
                switch.to_tid,
                switch.from_user_fs_base,
                switch.to_user_fs_base,
            );
        }
    }

    rt.irq_restore(irq);
}

pub fn resched_if_needed<R: BootRuntime>() {
    // Explicit safe-point check: yield if need_resched is set, but do NOT
    // run tick bookkeeping or decrement timeslices.
    let rt = crate::runtime::<R>();
    let irq = rt.irq_disable();

    let switch_params = {
        let lock = crate::sched::SCHEDULER.lock();
        if let Some(ptr) = *lock {
            let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
            sched.schedule_point(crate::sched::ScheduleReason::ReschedIfNeeded)
        } else {
            None
        }
    };

    if let Some(switch) = switch_params {
        let cr3_before = rt.debug_active_aspace_root();

        rt.tasking().activate_address_space(switch.to_aspace);

        let _cr3_after = rt.debug_active_aspace_root();

        unsafe {
            rt.tasking().switch_with_tls(
                &mut *switch.from_ctx,
                &*switch.to_ctx,
                switch.to_tid,
                switch.from_user_fs_base,
                switch.to_user_fs_base,
            );
        }
    }

    rt.irq_restore(irq);
}

pub fn dump_stats<R: BootRuntime>() {
    crate::sched::dump_stats::<R>();
}

/// Bootstrap a CPU for scheduling.
///
/// Must be called before the first yield on any CPU that doesn't already
/// have a current thread set (e.g. secondary CPUs).
fn bootstrap_cpu<R: BootRuntime>() {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let cpu_idx = crate::sched::current_cpu_index::<R>();

    crate::kinfo!("SMP: bootstrap_cpu start on CPU {}", cpu_idx);

    let lock = crate::sched::SCHEDULER.lock();
    if let Some(ptr) = *lock {
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };

        if let Some(pc) = sched.state.per_cpu.get_mut(cpu_idx) {
            if pc.current.is_none() {
                // CPU hasn't been bootstrapped yet — set current to idle thread.
                if let Some(idle_id) = pc.idle_task {
                    pc.current = Some(idle_id);
                    rt.set_current_tid(idle_id);
                    sched.state.mark_cpu_online(cpu_idx);
                    crate::kinfo!(
                        "SMP: CPU {} bootstrapped with idle thread {} and is now schedulable",
                        cpu_idx,
                        idle_id
                    );

                    if let Some(mut t) = crate::task::registry::get_thread_mut::<R>(idle_id) {
                        t.state = ThreadState::Running;
                    }
                } else {
                    crate::kerror!("SMP: CPU {} has no idle thread!", cpu_idx);
                }
            } else {
                crate::kdebug!(
                    "SMP: CPU {} bootstrap_cpu saw existing current thread {:?}",
                    cpu_idx,
                    pc.current
                );
            }
        }
    }

    rt.irq_restore(_irq);
}

pub fn run_scheduler<R: BootRuntime>() -> ! {
    crate::kinfo!("SMP: run_scheduler entry on CPU {}", crate::sched::current_cpu_index::<R>());
    // Bootstrap this CPU if needed (sets current thread for secondary CPUs).
    bootstrap_cpu::<R>();

    // Enable interrupts so this CPU can be preempted or woken from idle (HLT).
    crate::runtime::<R>().irq_restore(crate::IrqState(1));

    let mut idle_count: u64 = 0;
    loop {
        if !yield_now::<R>() {
            // No runnable work — halt until next IRQ (timer tick, device, IPI).
            crate::runtime::<R>().wait_for_interrupt();
            crate::sched::DIAG_HLT_WAKE.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

            idle_count += 1;
            if idle_count % 1000 == 0 {
                let cpu = crate::sched::current_cpu_index::<R>();
                crate::ktrace!("SCHED: CPU {} idle pulse", cpu);
            }
        }
    }
}
