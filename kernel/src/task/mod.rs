pub mod bridge;
pub mod exec;
pub mod loader;
pub mod registry;
use crate::sched as scheduler;

pub use crate::sched::Scheduler;

use crate::BootRuntime;
use crate::BootTasking;
use crate::simd::SimdState;
use abi::types::StackInfo;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

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

/// First-class process object.
///
/// A `Process` is the unit of resource ownership in Thing-OS.  Every user
/// process has exactly one `Process`, shared via `Arc<Mutex<Process>>` by all
/// threads that belong to it.  Kernel-only threads have `process_info = None`.
///
/// # Ownership model
///
/// | Resource           | Owner   | How threads access it               |
/// |--------------------|---------|-------------------------------------|
/// | PID / PPID         | Process | `process.lock().pid`                |
/// | VM address space   | Process | `process.lock().aspace_raw`         |
/// | VM mappings        | Process | `process.lock().mappings` (Arc)     |
/// | FD table           | Process | `process.lock().fd_table`           |
/// | CWD                | Process | `process.lock().cwd`                |
/// | VFS namespace      | Process | `process.lock().namespace`          |
/// | argv / env / auxv  | Process | `process.lock().argv` etc.          |
/// | Thread list        | Process | `process.lock().thread_ids`         |
/// | exec path          | Process | `process.lock().exec_path`          |
///
/// # Locking rules
///
/// The `Process` mutex must **never** be acquired while the scheduler lock
/// (`SCHEDULER.lock()`) is held — reverse order causes deadlock.
///
/// See `docs/concepts/process-object.md` for the full design document.
pub struct Process {
    /// Thread Group ID — the PID of the thread-group leader.
    pub pid: u32,
    /// PID of the parent process.
    pub ppid: u32,
    /// Process group ID.
    pub pgid: u32,
    /// Session ID.
    pub sid: u32,
    /// True when this process is the leader of its session.
    pub session_leader: bool,
    /// Argument vector passed at spawn/exec time.
    pub argv: Vec<Vec<u8>>,
    /// Environment variables.
    pub env: BTreeMap<Vec<u8>, Vec<u8>>,
    /// ELF auxiliary vector (AT_* entries as `(type, value)` pairs).
    pub auxv: Vec<(u64, u64)>,
    /// File descriptor table — fds 0/1/2 pre-populated at spawn time.
    pub fd_table: crate::vfs::fd_table::FdTable,
    /// VFS namespace — currently a global stub shared by all processes.
    ///
    /// [`crate::vfs::NamespaceRef`] is a unit struct today: all instances
    /// resolve to the same underlying global mount table.  The field exists
    /// so that per-process namespace isolation can be added later without
    /// touching every spawn call site.
    ///
    /// See `docs/concepts/namespaces.md` for the behaviour matrix and roadmap.
    pub namespace: crate::vfs::NamespaceRef,
    /// Current working directory.
    pub cwd: alloc::string::String,
    /// TIDs of all threads in this thread group.
    ///
    /// The first entry is the thread-group leader (TID == PID).  Entries are
    /// added on `spawn_user_thread` and removed when a thread exits.
    pub thread_ids: Vec<ThreadId>,
    /// Set while an `exec` is in progress; blocks new `SYS_SPAWN_THREAD` calls.
    pub exec_in_progress: bool,
    /// Path of the currently-running executable image.
    pub exec_path: alloc::string::String,
    /// Process-scoped VM mapping list.
    ///
    /// Every `Thread` in this process holds a clone of this `Arc` in its own
    /// `mappings` field so the scheduler's hot path (the `CURRENT_MAPPINGS`
    /// per-CPU cache) works without locking the `Process` mutex on every
    /// context switch.  The underlying `MappingList` is therefore always the
    /// same object visible from both `Process.mappings` and each thread's
    /// `Thread.mappings`.
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
    /// Per-process signal state: dispositions, pending set, stop/alarm state.
    pub signals: crate::signal::ProcessSignals,
    /// Exited children waiting for `waitpid` to consume their status.
    ///
    /// Each entry is `(child_pid, wait_status)`.  The status is encoded in
    /// the same format as POSIX `waitpid`: normal exit uses `(code << 8)`,
    /// signal termination uses `signum`, and stopped/continued children use
    /// the appropriate `w_stop_sig` / `w_continued` values.
    pub children_done: alloc::collections::VecDeque<(u32, i32)>,
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

    /// VM mapping list — a clone of `Process.mappings` (same underlying `Arc`).
    ///
    /// Kept here for zero-lock fast access by the scheduler's per-CPU mapping
    /// cache (`CURRENT_MAPPINGS`).  Always updated atomically with
    /// `Process.mappings` during exec or thread creation.
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
    crate::kinfo!(
        "SMP: run_scheduler entry on CPU {}",
        crate::sched::current_cpu_index::<R>()
    );
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
