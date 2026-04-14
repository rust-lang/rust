//! Core scheduler types and data structures.

use crate::BootRuntime;
use crate::task::TaskId;
use core::marker::PhantomData;

/// Default time slice in ticks (~100ms at 100Hz timer)
pub const DEFAULT_TIMESLICE: u32 = 10;

/// Maximum number of CPUs supported
pub const MAX_CPUS: usize = 32;

/// Anti-starvation: ticks to wait before boosting priority by one level
/// At 100Hz, 500 ticks = ~5 seconds
///
/// This ensures low-priority tasks don't starve even when high-priority tasks
/// are continuously runnable. After waiting for AGING_THRESHOLD_TICKS, a task's
/// priority is temporarily boosted by one level until it gets scheduled.
pub const AGING_THRESHOLD_TICKS: u64 = 500;

/// Anti-starvation: maximum priority boost levels (prevents excessive boosting)
///
/// Limits how many priority levels a task can be boosted. For example, with
/// MAX_PRIORITY_BOOST = 2, a Low priority task can be boosted to at most High
/// priority (Low -> Normal -> High), but never to Realtime.
pub const MAX_PRIORITY_BOOST: usize = 2;

// PerCpu is now in state.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackFaultResult {
    NotStack,
    Grew,
    Overflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleReason {
    /// Timer ISR path only — runs tick bookkeeping (wake sleepers, watchdog,
    /// wait-time aging, timeslice decrement).
    PreemptTick,
    CooperativeYield,
    SleepWait,
    BlockedOnIo,
    /// Used by preempt_enable(). Yields if need_resched is set but does NOT
    /// run tick bookkeeping or decrement timeslices.
    SafePoint,
    /// Used by explicit resched_if_needed() checks at syscall-return or other
    /// safe points. Same behaviour as SafePoint, semantically distinct.
    ReschedIfNeeded,
}

pub struct SwitchParams<Ctx, AS> {
    pub from_ctx: *mut Ctx,
    pub to_ctx: *const Ctx,
    pub to_aspace: AS,
    pub from_aspace: AS,
    pub from_tid: TaskId,
    pub to_tid: TaskId,
    pub from_user: bool,
    pub to_user: bool,
    /// Pointer into the outgoing task's `user_fs_base` field (saved on switch-out).
    pub from_user_fs_base: *mut u64,
    /// The incoming task's saved TLS base (restored on switch-in).
    pub to_user_fs_base: u64,
}

pub(crate) struct SchedulerMetrics {
    pub yields: u64,
    pub pops: u64,
    pub pushes: u64,
    pub idle_picks: u64,
    pub last_flush: u64,
}

pub struct Scheduler<R: BootRuntime> {
    pub(crate) state: crate::sched::state::SchedState,
    pub(crate) next_id: TaskId,
    pub(crate) preempt_disable_depth: usize,
    pub(crate) preempt_disable_since: u64,
    pub(crate) watchdog_warned: bool,
    pub(crate) total_cpu_count: usize,
    pub(crate) bringup_in_progress: bool,
    pub(crate) metrics: SchedulerMetrics,
    _phantom: core::marker::PhantomData<R>,
}

impl SchedulerMetrics {
    pub fn new() -> Self {
        SchedulerMetrics {
            yields: 0,
            pops: 0,
            pushes: 0,
            idle_picks: 0,
            last_flush: 0,
        }
    }
}

impl<R: BootRuntime> Scheduler<R> {
    pub fn new() -> Self {
        Scheduler {
            state: crate::sched::state::SchedState::new(),
            next_id: 1,
            preempt_disable_depth: 0,
            preempt_disable_since: 0,
            watchdog_warned: false,
            total_cpu_count: 1,
            bringup_in_progress: false,
            metrics: SchedulerMetrics::new(),
            _phantom: PhantomData,
        }
    }

    pub fn current_id(&self) -> Option<TaskId> {
        // This is tricky without knowing which CPU we are asking about.
        // For backwards compat logging, valid use mainly inside scheduler or per-cpu hooks.
        // We really need current_cpu_index here.
        // But Scheduler::current_id passed no index.
        // We will return None or rely on caller to use per-cpu accessors.
        // Actually, let's remove this helper or make it panic/useless?
        // Or better: `Scheduler` methods should generally task `cpu_index`?
        None
    }

    pub fn current_id_on_cpu(&self, cpu: usize) -> Option<TaskId> {
        self.state.per_cpu.get(cpu).and_then(|pc| pc.current)
    }

    pub fn current_priority(&self) -> Option<crate::task::TaskPriority> {
        // Also needs cpu index.
        None
    }

    pub fn current_priority_on_cpu(&self, cpu: usize) -> Option<crate::task::TaskPriority> {
        let tid = self.current_id_on_cpu(cpu)?;
        crate::task::registry::get_task::<R>(tid).map(|t| t.priority)
    }

    /// Returns `true` if there is at least one task in a non-idle run queue
    /// (priority levels 1–4) for the given CPU. Used by `run_scheduler` to
    /// decide whether to halt or keep spinning.
    pub fn has_runnable_work(&self, cpu_idx: usize) -> bool {
        if let Some(pc) = self.state.per_cpu.get(cpu_idx) {
            // Check priority queues 1 (Low) through 4 (Realtime)
            pc.runq[1..].iter().any(|q| !q.is_empty())
        } else {
            false
        }
    }
}
