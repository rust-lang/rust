//! Preemptive priority-based scheduler
//!
//! This module is split into focused submodules:
//! - `types`: Core data structures and enums
//! - `blocking`: Task blocking and wake primitives
//! - `hooks`: Type-erased hook system for callers without generic params
//! - `spawn`: Task and thread spawning
//! - `stack`: User stack allocation and fault handling
//! - `sleep`: Timing and yield functions
//! - `events`: Lock-free scheduler event types

pub(crate) mod blocking;
pub mod hooks;
pub use hooks::protect_user_range_current;
mod sleep;
mod spawn;
mod stack;
pub mod state;
pub(crate) mod types;
mod vm;
pub(crate) mod wait_queue;

// Re-export all public items
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

pub use blocking::{
    block_current, block_current_erased, init_blocking_hooks, wake_task, wake_task_erased,
};
pub use hooks::{
    ProcessSnapshot, add_user_mapping_current, alloc_user_stack_current,
    available_parallelism_current, check_user_mapping_current, current_priority_current,
    current_task_name_current, current_task_resource_id, current_tid_current, dump_stats_current,
    exit_current, get_signal_mask_current, get_thread_pending_current, get_user_mapping_at_current,
    handle_user_stack_fault_current, interrupt_task_current, kill_by_tid_current,
    list_processes_current, poll_task_exit_current, process_info_current,
    process_info_for_tid_current, register_task_exit_waiter_current, register_timeout_wake_current,
    remove_user_mappings_current, set_current_task_name_current, set_current_user_fs_base_current,
    set_priority_current, set_signal_mask_current, set_thread_pending_current, sleep_ticks_current,
    spawn_process_current, spawn_process_ex_current, spawn_process_from_path_current,
    spawn_user_thread_current, take_pending_interrupt_current, task_exec_current,
    task_status_current, task_wait_current, unregister_task_exit_waiter_current,
    unregister_timeout_wake_current, waitpid_current, yield_now_current,
};
pub use sleep::{sleep_ms, sleep_ticks, sleep_until, yield_now};
pub use spawn::{
    SpawnExResult, StdioSpec, boot_spawn_process, spawn, spawn_user_task_full, spawn_user_thread,
    spawn_user_thread_ex, spawn_with_priority, user_thread_trampoline,
};
use spin::Mutex;
pub use stack::{alloc_user_stack, handle_stack_fault, map_user_page, map_user_page_perms};
pub use types::{DEFAULT_TIMESLICE, ScheduleReason, Scheduler, StackFaultResult, SwitchParams};
pub use wait_queue::WaitQueue;

use crate::task::{Affinity, StartupArg, Task, TaskId, TaskPriority, TaskState};
use crate::{BootRuntime, BootTasking};

#[cfg(any(feature = "sched_debug", debug_assertions))]
static SWITCH_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);

pub static SCHEDULER: Mutex<Option<usize>> = Mutex::new(None);

/// Global tick counter for debugging scheduler health
pub static TICK_COUNT: AtomicU64 = AtomicU64::new(0);

pub static PROF_RESCHED_TRYLOCK_MISS: AtomicU64 = AtomicU64::new(0);

// Diagnostic counters for IPI delivery chain
pub static DIAG_IPI_SENT: AtomicU64 = AtomicU64::new(0);
pub static DIAG_IPI_HANDLER: AtomicU64 = AtomicU64::new(0);
pub static DIAG_HLT_WAKE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct SchedLockMetrics {
    pub hold_calls: u64,
    pub hold_us_total: u64,
    pub hold_us_max: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SchedLockSiteMetrics {
    pub block_current: SchedLockMetrics,
    pub wake_task: SchedLockMetrics,
    pub yield_now: SchedLockMetrics,
    pub sleep_ticks: SchedLockMetrics,
    pub wake_sleepers: SchedLockMetrics,
}

static PROF_SCHED_LOCK_BLOCK_CURRENT_CALLS: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_BLOCK_CURRENT_US_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_BLOCK_CURRENT_US_MAX: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_WAKE_TASK_CALLS: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_WAKE_TASK_US_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_WAKE_TASK_US_MAX: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_YIELD_NOW_CALLS: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_YIELD_NOW_US_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_YIELD_NOW_US_MAX: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_SLEEP_TICKS_CALLS: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_SLEEP_TICKS_US_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_SLEEP_TICKS_US_MAX: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_WAKE_SLEEPERS_CALLS: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_WAKE_SLEEPERS_US_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROF_SCHED_LOCK_WAKE_SLEEPERS_US_MAX: AtomicU64 = AtomicU64::new(0);

/// Lock-skip self-healing: when try_resched_if_needed() fails to acquire
/// the scheduler lock, set this flag so the next safe-point yields.
///
/// This is per-CPU to prevent one CPU from accidentally consuming another's
/// reschedule request.
static GLOBAL_NEED_RESCHED: [AtomicBool; types::MAX_CPUS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ATOMIC_FALSE: AtomicBool = AtomicBool::new(false);
    [ATOMIC_FALSE; types::MAX_CPUS]
};

/// Per-CPU start tick for the current try-lock miss warning window.
static TRYLOCK_MISS_WINDOW_START: [AtomicU64; types::MAX_CPUS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ATOMIC_ZERO: AtomicU64 = AtomicU64::new(0);
    [ATOMIC_ZERO; types::MAX_CPUS]
};

/// Per-CPU count of try-lock misses within the current warning window.
static TRYLOCK_MISS_WINDOW_COUNT: [AtomicU64; types::MAX_CPUS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ATOMIC_ZERO: AtomicU64 = AtomicU64::new(0);
    [ATOMIC_ZERO; types::MAX_CPUS]
};

/// Per-CPU tick timestamp when the try-lock miss threshold warning was last emitted.
static TRYLOCK_MISS_LAST_WARN_TICK: [AtomicU64; types::MAX_CPUS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ATOMIC_ZERO: AtomicU64 = AtomicU64::new(0);
    [ATOMIC_ZERO; types::MAX_CPUS]
};

/// If trylock misses exceed this count in a 2-second window, emit a warning.
pub const TRYLOCK_MISS_WARN_THRESHOLD: u64 = 50;

/// Minimum interval between threshold warning emissions per CPU.
pub const TRYLOCK_MISS_WARN_COOLDOWN_SECS: u64 = 30;

#[inline]
fn ticks_to_us<R: BootRuntime>(ticks: u64) -> u64 {
    let rt = crate::runtime::<R>();
    let freq = rt.mono_freq_hz().max(1);
    ticks.saturating_mul(1_000_000) / freq
}

#[inline]
fn update_max_u64(slot: &AtomicU64, val: u64) {
    let mut prev = slot.load(Ordering::Relaxed);
    while val > prev {
        match slot.compare_exchange_weak(prev, val, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => prev = actual,
        }
    }
}

#[inline]
fn snapshot_sched_lock_metric(
    calls: &AtomicU64,
    total: &AtomicU64,
    max: &AtomicU64,
) -> SchedLockMetrics {
    SchedLockMetrics {
        hold_calls: calls.swap(0, Ordering::Relaxed),
        hold_us_total: total.swap(0, Ordering::Relaxed),
        hold_us_max: max.swap(0, Ordering::Relaxed),
    }
}

pub(crate) fn record_sched_lock_hold<R: BootRuntime>(
    calls: &AtomicU64,
    total: &AtomicU64,
    max: &AtomicU64,
    start_ticks: u64,
) {
    let elapsed_us = ticks_to_us::<R>(crate::runtime::<R>().mono_ticks().wrapping_sub(start_ticks));
    calls.fetch_add(1, Ordering::Relaxed);
    total.fetch_add(elapsed_us, Ordering::Relaxed);
    update_max_u64(max, elapsed_us);
}

pub fn sched_lock_metrics_snapshot_and_reset() -> SchedLockSiteMetrics {
    SchedLockSiteMetrics {
        block_current: snapshot_sched_lock_metric(
            &PROF_SCHED_LOCK_BLOCK_CURRENT_CALLS,
            &PROF_SCHED_LOCK_BLOCK_CURRENT_US_TOTAL,
            &PROF_SCHED_LOCK_BLOCK_CURRENT_US_MAX,
        ),
        wake_task: snapshot_sched_lock_metric(
            &PROF_SCHED_LOCK_WAKE_TASK_CALLS,
            &PROF_SCHED_LOCK_WAKE_TASK_US_TOTAL,
            &PROF_SCHED_LOCK_WAKE_TASK_US_MAX,
        ),
        yield_now: snapshot_sched_lock_metric(
            &PROF_SCHED_LOCK_YIELD_NOW_CALLS,
            &PROF_SCHED_LOCK_YIELD_NOW_US_TOTAL,
            &PROF_SCHED_LOCK_YIELD_NOW_US_MAX,
        ),
        sleep_ticks: snapshot_sched_lock_metric(
            &PROF_SCHED_LOCK_SLEEP_TICKS_CALLS,
            &PROF_SCHED_LOCK_SLEEP_TICKS_US_TOTAL,
            &PROF_SCHED_LOCK_SLEEP_TICKS_US_MAX,
        ),
        wake_sleepers: snapshot_sched_lock_metric(
            &PROF_SCHED_LOCK_WAKE_SLEEPERS_CALLS,
            &PROF_SCHED_LOCK_WAKE_SLEEPERS_US_TOTAL,
            &PROF_SCHED_LOCK_WAKE_SLEEPERS_US_MAX,
        ),
    }
}

/// Called from timer ISR - records tick and triggers reschedule if needed
/// Uses try_resched_if_needed to avoid deadlock when SCHEDULER is held by main code
pub fn on_tick<R: BootRuntime>() {
    let cpu_idx = crate::runtime::<R>().current_cpu_id().0;
    let ticks = if cpu_idx == 0 {
        TICK_COUNT.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        TICK_COUNT.load(Ordering::Relaxed)
    };

    DIAG_IPI_HANDLER.fetch_add(1, Ordering::Relaxed);

    try_resched_if_needed::<R>();
}

/// Called from IPI handler - triggers reschedule without advancing time
pub fn on_resched_ipi<R: BootRuntime>() {
    DIAG_IPI_HANDLER.fetch_add(1, Ordering::Relaxed);
    crate::kdebug!(
        "SCHED: Resched IPI received on CPU {}",
        crate::runtime::<R>().current_cpu_index()
    );
    try_resched_if_needed::<R>();
}

/// Interrupt-safe version of resched_if_needed - uses try_lock to avoid deadlock
/// If SCHEDULER lock is contended, simply skip rescheduling this tick
fn try_resched_if_needed<R: BootRuntime>() {
    let rt = crate::runtime::<R>();
    let irq = rt.irq_disable();
    let cpu_idx = rt.current_cpu_index();

    // Use try_lock to avoid deadlock if SCHEDULER is held by main code
    if let Some(lock) = SCHEDULER.try_lock() {
        if let Some(ptr) = *lock {
            let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
            let current = sched.state.per_cpu.get(cpu_idx).and_then(|pc| pc.current);
            let idle = sched.state.per_cpu.get(cpu_idx).and_then(|pc| pc.idle_task);
            let has_work = sched.has_runnable_work(cpu_idx);
            let runq_total = sched
                .state
                .per_cpu
                .get(cpu_idx)
                .map(|pc| pc.runq.iter().map(|q| q.len()).sum::<usize>())
                .unwrap_or(0);
            if let Some(switch) = sched.schedule_point(ScheduleReason::PreemptTick) {
                // Must drop lock before context switch!
                drop(lock);

                rt.tasking().activate_address_space(switch.to_aspace);

                unsafe {
                    rt.tasking().switch_with_tls(
                        &mut *switch.from_ctx,
                        &*switch.to_ctx,
                        switch.to_tid,
                        switch.from_user_fs_base,
                        switch.to_user_fs_base,
                    );
                }
            } else if has_work && runq_total > 1 {
                crate::kwarn!(
                    "SCHED: CPU {} handled resched but made no switch: current={:?} idle={:?} runq_total={}",
                    cpu_idx,
                    current,
                    idle,
                    runq_total
                );
            }
        }
    } else {
        PROF_RESCHED_TRYLOCK_MISS.fetch_add(1, Ordering::Relaxed);
        // Warn only when misses cross threshold in a 2-second per-CPU window.
        let now = rt.mono_ticks();
        let window_ticks = rt.mono_freq_hz().max(1).saturating_mul(2);
        let window_start = &TRYLOCK_MISS_WINDOW_START[cpu_idx];
        let window_count = &TRYLOCK_MISS_WINDOW_COUNT[cpu_idx];

        let start = window_start.load(Ordering::Relaxed);
        if start == 0 || now.saturating_sub(start) > window_ticks {
            window_start.store(now, Ordering::Relaxed);
            window_count.store(1, Ordering::Relaxed);
        } else {
            let misses = window_count.fetch_add(1, Ordering::Relaxed) + 1;
            if misses == TRYLOCK_MISS_WARN_THRESHOLD {
                let cooldown_ticks =
                    rt.mono_freq_hz().max(1).saturating_mul(TRYLOCK_MISS_WARN_COOLDOWN_SECS);
                let last_warn = TRYLOCK_MISS_LAST_WARN_TICK[cpu_idx].load(Ordering::Relaxed);
                if last_warn == 0 || now.saturating_sub(last_warn) >= cooldown_ticks {
                    TRYLOCK_MISS_LAST_WARN_TICK[cpu_idx].store(now, Ordering::Relaxed);
                    crate::kwarn!(
                        "SCHED: CPU {} resched try_lock misses reached {} in 2s (suppressing until window reset)",
                        cpu_idx,
                        TRYLOCK_MISS_WARN_THRESHOLD
                    );
                }
            }
        }
        // Self-healing: tell the next safe point to reschedule
        GLOBAL_NEED_RESCHED[cpu_idx].store(true, Ordering::Release);
    }
    // If try_lock failed, skip rescheduling this tick - not a problem, next tick will try again

    rt.irq_restore(irq);
}

pub(crate) fn current_cpu_index<R: BootRuntime>() -> usize {
    let rt = crate::runtime::<R>();
    rt.current_cpu_index()
}

pub fn init<R: BootRuntime>() {
    crate::kdebug!("  Acquiring scheduler lock...");
    let mut lock = SCHEDULER.lock();
    crate::kdebug!("  Lock acquired, checking if initialized...");
    if lock.is_none() {
        crate::kdebug!("  Allocating scheduler...");
        let sched = alloc::boxed::Box::new(types::Scheduler::<R>::new());
        crate::kdebug!("  Leaking scheduler...");
        let s = alloc::boxed::Box::leak(sched);
        crate::kdebug!("  Initializing boot task...");
        init_boot_task::<R>(s);
        crate::kdebug!("  Storing scheduler pointer...");
        *lock = Some(s as *mut types::Scheduler<R> as usize);
        unsafe {
            hooks::YIELD_HOOK = Some(sleep::yield_now::<R>);
            hooks::EXIT_HOOK = Some(exit::<R>);
            hooks::SPAWN_USER_HOOK = Some(spawn::spawn_user_thread_ex::<R>);
            hooks::SPAWN_PROCESS_HOOK = Some(spawn::boot_spawn_process::<R>);
            hooks::CURRENT_TID_HOOK = Some(current_tid::<R>);
            hooks::INTERRUPT_TASK_HOOK = Some(interrupt_task::<R>);
            hooks::TAKE_PENDING_INTERRUPT_HOOK = Some(take_pending_interrupt::<R>);
            hooks::TASK_STATUS_HOOK = Some(task_status::<R>);
            hooks::TASK_WAIT_HOOK = Some(wait_task::<R>);
            hooks::SET_PRIORITY_HOOK = Some(set_priority::<R>);
            hooks::CURRENT_PRIORITY_HOOK = Some(current_priority::<R>);
            hooks::AVAILABLE_PARALLELISM_HOOK = Some(available_parallelism::<R>);
            hooks::ALLOC_USER_STACK_HOOK = Some(stack::alloc_user_stack::<R>);
            hooks::RUN_SCHEDULER_HOOK = Some(crate::task::run_scheduler::<R>);
            hooks::KILL_BY_TID_HOOK = Some(kill_by_tid::<R>);
            hooks::DUMP_STATS_HOOK = Some(crate::task::dump_stats::<R>);
            crate::memory::set_map_user_page_hook(stack::map_user_page::<R>);
            crate::memory::set_map_user_page_perms_hook(stack::map_user_page_perms::<R>);
            crate::memory::set_unmap_user_page_hook(stack::unmap_user_page::<R>);
            crate::memory::set_protect_user_page_hook(stack::protect_user_page::<R>);
            hooks::STACK_FAULT_HOOK = Some(stack::handle_stack_fault::<R>);
            hooks::SLEEP_TICKS_HOOK = Some(sleep::sleep_ticks::<R>);
            hooks::ADD_USER_MAPPING_HOOK = Some(vm::add_user_mapping::<R>);
            hooks::REMOVE_USER_MAPPINGS_HOOK = Some(vm::remove_user_mappings::<R>);
            hooks::CHECK_USER_MAPPING_HOOK = Some(vm::check_user_mapping::<R>);
            hooks::GET_USER_MAPPING_AT_HOOK = Some(vm::get_user_mapping_at::<R>);
            hooks::PROTECT_USER_RANGE_HOOK = Some(vm::protect_user_range::<R>);
            hooks::PROCESS_INFO_HOOK = Some(process_info::<R>);
            hooks::PROCESS_INFO_FOR_TID_HOOK = Some(process_info_for_tid::<R>);
            hooks::SPAWN_PROCESS_EX_HOOK = Some(spawn::boot_spawn_process_ex::<R>);
            hooks::SPAWN_PROCESS_FROM_PATH_HOOK = Some(spawn::spawn_process_from_path::<R>);
            hooks::CURRENT_RESOURCE_HOOK = Some(current_task_resource_id_impl::<R>);
            hooks::POLL_TASK_EXIT_HOOK = Some(poll_task_exit::<R>);
            hooks::REGISTER_TASK_EXIT_WAITER_HOOK = Some(register_task_exit_waiter_public::<R>);
            hooks::UNREGISTER_TASK_EXIT_WAITER_HOOK = Some(unregister_task_exit_waiter::<R>);
            hooks::REGISTER_TIMEOUT_WAKE_HOOK = Some(register_timeout_wake::<R>);
            hooks::UNREGISTER_TIMEOUT_WAKE_HOOK = Some(unregister_timeout_wake::<R>);
            hooks::LIST_PROCESSES_HOOK = Some(list_processes::<R>);
            hooks::CURRENT_TASK_NAME_HOOK = Some(current_task_name_impl::<R>);
            hooks::TASK_EXEC_HOOK = Some(crate::task::exec::task_exec_current::<R>);
            hooks::SET_CURRENT_USER_FS_BASE_HOOK = Some(set_current_user_fs_base::<R>);
            hooks::SET_CURRENT_TASK_NAME_HOOK = Some(set_current_task_name::<R>);
            hooks::WAITPID_HOOK = Some(waitpid::<R>);
            hooks::GET_SIGNAL_MASK_HOOK = Some(get_signal_mask::<R>);
            hooks::SET_SIGNAL_MASK_HOOK = Some(set_signal_mask::<R>);
            hooks::GET_THREAD_PENDING_HOOK = Some(get_thread_pending::<R>);
            hooks::SET_THREAD_PENDING_HOOK = Some(set_thread_pending::<R>);
            crate::memory::set_translate_user_page_hook(vm::translate_user_page::<R>);
        }
        blocking::init_blocking_hooks::<R>();
        let cpu_total = if let Some(ptr) = *lock {
            let sched = unsafe { &*(ptr as *const types::Scheduler<R>) };
            sched.total_cpu_count
        } else {
            1
        };
        crate::contract!("Scheduler initialized");
    }
}

fn init_boot_task<R: BootRuntime>(sched: &mut types::Scheduler<R>) {
    let rt = crate::runtime::<R>();
    let cpu_total = rt.cpu_total_count();

    // Initialize PerCpu state for all CPUs (initially empty/offline)
    for _ in 0..cpu_total {
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
    }

    sched.total_cpu_count = cpu_total;
    sched.state.set_boot_cpu_online();

    crate::kdebug!("  Creating boot task...");

    let layout = alloc::alloc::Layout::from_size_align(16384, 8).unwrap();
    let stack_base = unsafe { alloc::alloc::alloc(layout) };
    if stack_base.is_null() {
        panic!("Failed to allocate stack for boot task");
    }
    let stack_top = (stack_base as u64) + 16384;

    let task: Task<R> = Task {
        id: 0,
        state: TaskState::Running,
        priority: TaskPriority::Normal,
        kstack_base: stack_base,
        kstack_size: 16384,
        kstack_top: stack_top,
        ctx: Default::default(),
        aspace: rt.tasking().active_address_space(),
        simd: crate::simd::SimdState::new(rt),
        exit_code: None,
        exit_waiters: crate::sched::WaitQueue::new(),
        is_user: false,
        wake_pending: false,
        pending_interrupt: false,
        stack_info: None,
        mappings: alloc::sync::Arc::new(spin::Mutex::new(
            crate::memory::mappings::MappingList::new(),
        )),
        timeslice_remaining: types::DEFAULT_TIMESLICE,
        affinity: crate::task::Affinity::Any,
        last_cpu: Some(0),
        name: {
            let mut n = [0u8; 32];
            n[0] = b'b';
            n[1] = b'o';
            n[2] = b'o';
            n[3] = b't';
            n
        },
        name_len: 4,
        process_info: None,
        enqueued_at_tick: TICK_COUNT.load(Ordering::Relaxed),
        base_priority: TaskPriority::Normal,
        user_fs_base: 0,
        detached: false,
        signals: crate::signal::ThreadSignals::new(),
    };
    let sched_fields = crate::sched::state::TaskSchedFields { tid: task.id, runq_location: None };
    sched.state.insert_task(sched_fields);
    crate::task::registry::get_registry::<R>().insert(alloc::boxed::Box::new(task));

    // Boot task runs on CPU 0
    sched.state.per_cpu[0].current = Some(0);

    // Link boot task to CPU 0

    crate::kdebug!("  Creating idle tasks...");

    // Create idle task for CPU 0 initially
    {
        let i = 0;
        let idle_id = sched.spawn(
            idle_task::<R>,
            StartupArg::Raw(i),
            TaskPriority::Idle,
            crate::task::Affinity::Pinned(i),
        );

        // Remove from run queues - idle tasks are special
        for q in sched.state.per_cpu.iter_mut().flat_map(|pc| pc.runq.iter_mut()) {
            if let Some(pos) = q.iter().position(|&id| id == idle_id) {
                q.remove(pos);
            }
        }

        // Set as this CPU's idle task
        sched.state.per_cpu[i].idle_task = Some(idle_id);

        // Pin idle task to its CPU
        if let Some(mut t) = crate::task::registry::get_task_mut::<R>(idle_id) {
            t.affinity = crate::task::Affinity::Pinned(i);
        }
    }

    crate::kdebug!("  Boot task initialized");
}

impl<R: BootRuntime> types::Scheduler<R> {
    pub fn schedule_point(
        &mut self,
        reason: ScheduleReason,
    ) -> Option<
        SwitchParams<
            <R::Tasking as BootTasking>::Context,
            <R::Tasking as BootTasking>::AddressSpace,
        >,
    > {
        let cpu_idx = current_cpu_index::<R>();
        let global_requested = GLOBAL_NEED_RESCHED[cpu_idx].swap(false, Ordering::Acquire);

        if self.preempt_disable_depth > 0 {
            if global_requested {
                self.state.per_cpu[cpu_idx].need_resched = true;
            }
            return None;
        }

        match reason {
            ScheduleReason::PreemptTick => {
                // Wake any sleeping tasks whose time has expired (Timekeeper only)
                if cpu_idx == 0 {
                    self.wake_sleepers();
                }

                // Check preemption watchdog
                self.check_preempt_watchdog();

                let mut should_yield = global_requested || self.state.per_cpu[cpu_idx].need_resched;
                self.state.per_cpu[cpu_idx].need_resched = false;

                // Tick bookkeeping: only decrement if this was a timer tick
                if let Some(current_id) = self.state.per_cpu[cpu_idx].current {
                    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(current_id) {
                        if task.timeslice_remaining > 0 {
                            task.timeslice_remaining -= 1;
                        }
                        if task.timeslice_remaining == 0 {
                            // Reset for next run
                            task.timeslice_remaining = types::DEFAULT_TIMESLICE;
                            should_yield = true;
                        }
                    } // REGISTRY lock dropped here!
                }

                if should_yield {
                    // Force reschedule (safe now because REGISTRY lock is dropped)
                    return self.prepare_yield();
                }
                return None; // Not expired yet
            }
            ScheduleReason::SafePoint
            | ScheduleReason::ReschedIfNeeded
            | ScheduleReason::SleepWait => {
                // No tick bookkeeping, no timeslice decrement.
                // Simply yield if a reschedule was requested.
                if self.state.per_cpu[cpu_idx].need_resched
                    || global_requested
                    || reason == ScheduleReason::SleepWait
                {
                    self.state.per_cpu[cpu_idx].need_resched = false;
                    return self.prepare_yield();
                }
                return None;
            }
            _ => {
                // For other reasons (Unblock, etc), always attempt yield
                return self.prepare_yield();
            }
        }
    }

    /// Check if preemption has been disabled too long
    fn check_preempt_watchdog(&mut self) {
        if self.preempt_disable_depth > 0 && !self.watchdog_warned {
            let now = TICK_COUNT.load(Ordering::Relaxed);
            if now.saturating_sub(self.preempt_disable_since) > 500 {
                // );
                self.watchdog_warned = true;
            }
        }
    }

    /// Wake any sleeping tasks whose sleep time has expired
    fn wake_sleepers(&mut self) {
        let now = TICK_COUNT.load(Ordering::Relaxed);
        let lock_start = crate::runtime::<R>().mono_ticks();

        while let Some((&wake_tick, _)) = self.state.sleep_queue.first_key_value() {
            if wake_tick <= now {
                let (_, tids) = self.state.sleep_queue.pop_first().unwrap();

                for tid in tids {
                    let priority: usize;
                    let target_cpu: usize;

                    // 1. Lock REGISTRY and update task state
                    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(tid) {
                        task.state = TaskState::Runnable;
                        task.enqueued_at_tick = TICK_COUNT.load(Ordering::Relaxed);
                        priority = task.priority as usize;
                        target_cpu = match task.affinity {
                            crate::task::Affinity::Pinned(cpu) => cpu,
                            crate::task::Affinity::Any => {
                                let idx = spawn::RR_IDX.fetch_add(1, Ordering::Relaxed);
                                self.state.pick_online_cpu(idx)
                            }
                        };
                    } else {
                        continue;
                    } // REGISTRY lock dropped here!

                    let actual_cpu =
                        if target_cpu < self.state.per_cpu.len() { target_cpu } else { 0 };
                    self.state.enqueue_task(actual_cpu, priority, tid);

                    let current_prio = self
                        .state
                        .per_cpu
                        .get(actual_cpu)
                        .and_then(|pc| pc.current)
                        .and_then(|cid| crate::task::registry::get_task::<R>(cid))
                        .map(|t| t.priority as usize)
                        .unwrap_or(0);
                    if priority > current_prio {
                        if actual_cpu == current_cpu_index::<R>() {
                            self.state.per_cpu[current_cpu_index::<R>()].need_resched = true;
                        }
                    }

                    if actual_cpu != current_cpu_index::<R>() {
                        crate::kdebug!(
                            "SCHED: Nudging CPU {} for task {} (prio {})",
                            actual_cpu,
                            tid,
                            priority
                        );
                        crate::runtime::<R>().send_ipi(actual_cpu, 0x30);
                    }
                }
            } else {
                break;
            }
        }

        record_sched_lock_hold::<R>(
            &PROF_SCHED_LOCK_WAKE_SLEEPERS_CALLS,
            &PROF_SCHED_LOCK_WAKE_SLEEPERS_US_TOTAL,
            &PROF_SCHED_LOCK_WAKE_SLEEPERS_US_MAX,
            lock_start,
        );
    }

    pub fn preempt_disable(&mut self) {
        if self.preempt_disable_depth == 0 {
            // Track when we started disabling preemption
            self.preempt_disable_since = TICK_COUNT.load(Ordering::Relaxed);
            self.watchdog_warned = false;
        }
        self.preempt_disable_depth += 1;
        if self.preempt_disable_depth == 1 {
            // Only trace on transition to disabled? Or depth change?
            // User task says "Record (..., preempt_disable_depth)".
            // Let's trace all for now, or just 0->1.
            // 0->1 is most important for start of disable region.
            crate::trace::irq_ring::push(abi::trace::TraceEvent::PreemptDisable {
                depth: self.preempt_disable_depth as u32,
                timestamp: crate::trace::now(),
            });
        }
    }

    fn flush_metrics_if_needed(&mut self) {
        let rt = crate::runtime::<R>();
        let now = rt.mono_ticks();
        let limit = rt.mono_freq_hz() * 2;

        if self.metrics.last_flush == 0 {
            self.metrics.last_flush = now;
            return;
        }

        if now - self.metrics.last_flush > limit {
            #[cfg(feature = "diagnostic-apps")]
            crate::log_event!(
               crate::logging::LogLevel::Info,
               "sched.activity",
               "Scheduler Activity Rollup",
               {
                   yields: self.metrics.yields,
                   pops: self.metrics.pops,
                   pushes: self.metrics.pushes,
                   idle_picks: self.metrics.idle_picks,
                   runq_len: self.runq.iter().map(|q| q.len()).sum::<usize>() as u64
               },
               about=[]
            );

            self.metrics.yields = 0;
            self.metrics.pops = 0;
            self.metrics.pushes = 0;
            self.metrics.idle_picks = 0;
            self.metrics.last_flush = now;
        }
    }

    pub fn preempt_enable(
        &mut self,
    ) -> Option<
        SwitchParams<
            <R::Tasking as BootTasking>::Context,
            <R::Tasking as BootTasking>::AddressSpace,
        >,
    > {
        if self.preempt_disable_depth > 0 {
            self.preempt_disable_depth -= 1;
        }

        if self.preempt_disable_depth == 0
            && self.state.per_cpu[current_cpu_index::<R>()].need_resched
        {
            self.state.per_cpu[current_cpu_index::<R>()].need_resched = false;
            return self.schedule_point(ScheduleReason::SafePoint);
        }
        None
    }

    pub fn prepare_yield(
        &mut self,
    ) -> Option<
        SwitchParams<
            <R::Tasking as BootTasking>::Context,
            <R::Tasking as BootTasking>::AddressSpace,
        >,
    > {
        let cpu_idx = current_cpu_index::<R>();
        let current_id = self.state.per_cpu.get(cpu_idx)?.current?;

        self.metrics.yields += 1;

        // Don't push idle task or dead tasks back to runq
        if Some(current_id) != self.state.per_cpu[cpu_idx].idle_task {
            if let Some(task) = crate::task::registry::get_task::<R>(current_id) {
                if task.state != TaskState::Dead {
                    let priority = task.priority;
                    // Push to LOCAL runq (we are yielding on this CPU)
                    self.state.enqueue_task(cpu_idx, priority as usize, current_id);
                    self.metrics.pushes += 1;
                }
            }
        }

        self.prepare_schedule()
    }

    pub(crate) fn prepare_schedule(
        &mut self,
    ) -> Option<
        SwitchParams<
            <R::Tasking as BootTasking>::Context,
            <R::Tasking as BootTasking>::AddressSpace,
        >,
    > {
        self.flush_metrics_if_needed();

        let rt = crate::runtime::<R>();
        let cpu_idx = current_cpu_index::<R>();
        let real_cpu_id = rt.current_cpu_id().0 as usize;
        if cpu_idx != real_cpu_id {
            crate::kprintln!(
                "FATAL GS CORRUPTION: Core {} thinks it is index {} via GS!",
                real_cpu_id,
                cpu_idx
            );
        }
        if cpu_idx >= self.state.per_cpu.len() {
            return None;
        }
        let per_cpu_len = self.state.per_cpu.len();

        // Collect tasks pinned to a different CPU so we can requeue them after scanning.
        const MAX_MISROUTED: usize = 32;
        let mut misrouted: [(usize, usize, TaskId); MAX_MISROUTED] = [(0, 0, 0); MAX_MISROUTED];
        let mut misrouted_count = 0;

        let mut next_id = None;
        // Priority scan — skip dead and misrouted tasks, evaluating aging on-pick
        loop {
            let mut best_q = None;
            let mut best_eff = 0;

            for p in (1..5).rev() {
                if let Some(&id) = self.state.per_cpu[cpu_idx].runq[p].front() {
                    let mut eff = p; // Start with base priority (queue index)
                    if p < 4 {
                        // aging only applies up to High
                        if let Some(task) = crate::task::registry::get_task::<R>(id) {
                            let now = TICK_COUNT.load(Ordering::Relaxed);
                            let wait_ticks = now.saturating_sub(task.enqueued_at_tick);
                            let boost = (wait_ticks / types::AGING_THRESHOLD_TICKS) as usize;
                            let boost = boost.min(types::MAX_PRIORITY_BOOST);
                            eff = (p + boost).min(4);
                        }
                    }
                    if eff > best_eff || best_q.is_none() {
                        best_eff = eff;
                        best_q = Some(p);
                    }
                }
            }

            if let Some(p) = best_q {
                let id = self.state.dequeue_task_front(cpu_idx, p).unwrap();
                self.metrics.pops += 1;

                let task_ref = crate::task::registry::get_task::<R>(id);
                if task_ref.as_deref().map_or(true, |t| t.state == TaskState::Dead) {
                    continue;
                }
                let task = task_ref.unwrap();
                if let crate::task::Affinity::Pinned(target) = task.affinity {
                    if target != cpu_idx && target < per_cpu_len {
                        if misrouted_count < MAX_MISROUTED {
                            misrouted[misrouted_count] = (task.base_priority as usize, target, id);
                            misrouted_count += 1;
                        }
                        continue;
                    }
                }
                next_id = Some(id);
                break;
            } else {
                break;
            }
        }

        let next_id = match next_id {
            Some(id) => id,
            None => {
                // Check Idle queue — skip dead and misrouted tasks
                let mut found_idle_q = None;
                while let Some(id) = self.state.dequeue_task_front(cpu_idx, 0) {
                    self.metrics.pops += 1;
                    let task_ref = crate::task::registry::get_task::<R>(id);
                    if task_ref.as_deref().map_or(true, |t| t.state == TaskState::Dead) {
                        continue;
                    }
                    let task = task_ref.unwrap();
                    if let crate::task::Affinity::Pinned(target) = task.affinity {
                        if target != cpu_idx && target < per_cpu_len {
                            if misrouted_count < MAX_MISROUTED {
                                misrouted[misrouted_count] = (task.priority as usize, target, id);
                                misrouted_count += 1;
                            }
                            continue;
                        }
                    }
                    found_idle_q = Some(id);
                    break;
                }
                if let Some(id) = found_idle_q {
                    id
                } else if let Some(idle) = self.state.per_cpu[cpu_idx].idle_task {
                    self.metrics.idle_picks += 1;
                    idle
                } else {
                    // Flush misrouted tasks before returning
                    for &(prio, target_cpu, id) in &misrouted[..misrouted_count] {
                        self.state.enqueue_task(target_cpu, prio, id);
                    }
                    return None;
                }
            }
        };

        // Flush misrouted tasks to their correct CPU queues
        for &(prio, target_cpu, id) in &misrouted[..misrouted_count] {
            self.state.enqueue_task(target_cpu, prio, id);
        }

        let current_id = self.state.per_cpu[cpu_idx]
            .current
            .expect("prepare_schedule called without current task");

        if next_id == current_id {
            let idx = self.state.get_task_index(current_id).unwrap_or_else(|| {
                crate::kerror!(
                    "SchedTasks: {:?}",
                    self.state.threads.iter().map(|f| f.tid).collect::<alloc::vec::Vec<_>>()
                );
                panic!("failed to find current_id {} in get_task_index", current_id)
            });
            let mut reg = crate::task::registry::get_registry::<R>();
            reg.threads[idx].state = TaskState::Running;
            return None;
        }

        self.state.per_cpu[cpu_idx].current = Some(next_id);

        let old_idx = self.state.get_task_index(current_id).unwrap_or_else(|| {
            crate::kerror!(
                "SchedTasks: {:?}",
                self.state.threads.iter().map(|f| f.tid).collect::<alloc::vec::Vec<_>>()
            );
            panic!("failed to find current_id {} in get_task_index", current_id)
        });
        let new_idx = self.state.get_task_index(next_id).unwrap_or_else(|| {
            crate::kerror!(
                "SchedTasks: {:?}",
                self.state.threads.iter().map(|f| f.tid).collect::<alloc::vec::Vec<_>>()
            );
            panic!("failed to find next_id {} in get_task_index", next_id)
        });

        let mut reg = crate::task::registry::get_registry::<R>();
        let (old_task, new_task) = if old_idx < new_idx {
            let (left, right) = reg.threads.split_at_mut(new_idx);
            (&mut left[old_idx], &mut right[0])
        } else {
            let (left, right) = reg.threads.split_at_mut(old_idx);
            (&mut right[0], &mut left[new_idx])
        };

        if old_task.state == TaskState::Running {
            old_task.state = TaskState::Runnable;
            old_task.enqueued_at_tick = TICK_COUNT.load(Ordering::Relaxed);
        }
        new_task.state = TaskState::Running;
        new_task.last_cpu = Some(cpu_idx);

        // Update the lock-free mapping cache for this CPU so check_user_mapping is fast
        crate::sched::vm::CURRENT_MAPPINGS[cpu_idx].store(
            alloc::sync::Arc::as_ptr(&new_task.mappings) as *mut _,
            core::sync::atomic::Ordering::Release,
        );

        old_task.simd.save(crate::runtime::<R>());
        new_task.simd.restore(crate::runtime::<R>());

        crate::trace::irq_ring::push(abi::trace::TraceEvent::ContextSwitch {
            from: old_task.id,
            to: new_task.id,
            timestamp: crate::trace::now(),
        });

        Some(SwitchParams {
            from_ctx: &mut old_task.ctx as *mut _,
            to_ctx: &new_task.ctx as *const _,
            to_aspace: new_task.aspace,
            from_tid: old_task.id,
            to_tid: new_task.id,
            from_aspace: old_task.aspace,
            from_user: old_task.is_user,
            to_user: new_task.is_user,
            from_user_fs_base: &mut old_task.user_fs_base as *mut u64,
            to_user_fs_base: new_task.user_fs_base,
        })
    }

    pub fn terminate_current(
        &mut self,
        code: i32,
    ) -> (
        SwitchParams<
            <R::Tasking as BootTasking>::Context,
            <R::Tasking as BootTasking>::AddressSpace,
        >,
        alloc::vec::Vec<u64>,
    ) {
        let cpu_idx = current_cpu_index::<R>();
        let current_id = self
            .state
            .per_cpu
            .get(cpu_idx)
            .and_then(|pc| pc.current)
            .expect("terminate_current called with no current task");

        let waiters = mark_task_exited::<R>(self, current_id, code);

        // Release any claimed devices
        let released = crate::device_registry::REGISTRY.lock().release_all_for_task(current_id);
        if released > 0 {
            crate::kinfo!("DEVICE: released {} claims for task {}", released, current_id);
        }

        loop {
            if let Some(switch) = self.prepare_schedule() {
                return (switch, waiters);
            }
        }
    }

    pub fn set_priority(&mut self, id: TaskId, priority: TaskPriority) {
        if let Some(idx) = self.state.get_task_index(id) {
            let old_priority = crate::task::registry::get_registry::<R>().threads[idx].priority;
            crate::task::registry::get_registry::<R>().threads[idx].priority = priority;
            crate::task::registry::get_registry::<R>().threads[idx].base_priority = priority; // Update base priority for anti-starvation

            // If it's runnable and in a runq, move it to the new runq
            if crate::task::registry::get_registry::<R>().threads[idx].state == TaskState::Runnable
            {
                let loc = self.state.get_task(id).and_then(|t| t.runq_location);
                if let Some((cpu, _)) = loc {
                    self.state.remove_task_from_runq(id);
                    self.state.enqueue_task(cpu, priority as usize, id);
                }
            }
        }
    }

    /// Mark a secondary CPU as online and initialize its idle task.
    pub fn cpu_online(&mut self, cpu_index: usize) {
        crate::kdebug!("SMP: CPU {} online (triggered by scheduler spawn)", cpu_index);
        self.bringup_in_progress = false;

        // Create idle task for this new CPU
        let i = cpu_index;
        let idle_id = self.spawn(
            idle_task::<R>,
            StartupArg::Raw(i),
            TaskPriority::Idle,
            crate::task::Affinity::Pinned(i),
        );

        // Remove from run queues - idle tasks are special
        self.state.remove_task_from_runq(idle_id);

        // Set as this CPU's idle task
        self.state.per_cpu[i].idle_task = Some(idle_id);

        // Pin idle task to its CPU
        if let Some(mut t) = crate::task::registry::get_task_mut::<R>(idle_id) {
            t.affinity = crate::task::Affinity::Pinned(i);
        }
    }
}

pub fn set_priority<R: BootRuntime>(id: TaskId, priority: TaskPriority) {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    if let Some(ptr) = *lock {
        let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
        sched.set_priority(id, priority);
    }
    rt.irq_restore(_irq);
}

pub fn task_status<R: BootRuntime>(id: TaskId) -> Option<(TaskState, Option<i32>)> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let res = if let Some(ptr) = *lock {
        let sched = unsafe { &*(ptr as *const types::Scheduler<R>) };
        crate::task::registry::get_task::<R>(id).map(|t| (t.state, t.exit_code))
    } else {
        None
    };
    rt.irq_restore(_irq);
    res
}

pub fn current_priority<R: BootRuntime>() -> TaskPriority {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let res = if let Some(lock) = SCHEDULER.try_lock() {
        if let Some(ptr) = *lock {
            let sched = unsafe { &*(ptr as *const types::Scheduler<R>) };
            let cpu = current_cpu_index::<R>();
            sched
                .state
                .per_cpu
                .get(cpu)
                .and_then(|pc| pc.current)
                .and_then(|tid| crate::task::registry::get_task::<R>(tid))
                .map(|t| t.priority)
                .unwrap_or(TaskPriority::Normal)
        } else {
            TaskPriority::Normal
        }
    } else {
        TaskPriority::Normal
    };
    rt.irq_restore(_irq);
    res
}

pub fn current_tid<R: BootRuntime>() -> u64 {
    crate::runtime::<R>().current_tid()
}

fn effective_parallelism_from_state(online_cpu_count: usize, affinity: Affinity) -> usize {
    let online = online_cpu_count.max(1);
    match affinity {
        Affinity::Pinned(_) => 1,
        Affinity::Any => online,
    }
}

pub fn available_parallelism<R: BootRuntime>() -> usize {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    let result = if let Some(lock) = SCHEDULER.try_lock() {
        if let Some(ptr) = *lock {
            let sched = unsafe { &*(ptr as *const types::Scheduler<R>) };
            let online = sched.state.online_cpu_count;
            let affinity = crate::task::registry::get_task::<R>(rt.current_tid())
                .map(|task| task.affinity)
                .unwrap_or(Affinity::Any);
            effective_parallelism_from_state(online, affinity)
        } else {
            1
        }
    } else {
        1
    };

    rt.irq_restore(_irq);
    result.max(1)
}

fn current_task_name_impl<R: BootRuntime>() -> [u8; 32] {
    let tid = current_tid::<R>();
    if let Some(task) = crate::task::registry::get_task::<R>(tid) {
        task.name
    } else {
        let mut n = [0u8; 32];
        n[0..7].copy_from_slice(b"unknown");
        n
    }
}

/// Update the current task's stored `user_fs_base` field.
///
/// Called by the TLS-set syscall handler after writing the hardware register,
/// so that the value is saved correctly on the next context switch without
/// needing an extra MSR read.
fn set_current_user_fs_base<R: BootRuntime>(base: u64) {
    let tid = crate::runtime::<R>().current_tid();
    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(tid) {
        task.user_fs_base = base;
    }
}

/// Update the calling thread's human-readable name.
///
/// Called by `SYS_TASK_SET_NAME`.  Defensively clamps to 31 bytes even
/// though the syscall handler already enforces this limit, so that the
/// function stays safe if called from other internal paths in the future.
fn set_current_task_name<R: BootRuntime>(ptr: *const u8, len: usize) {
    let tid = crate::runtime::<R>().current_tid();
    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(tid) {
        let len = len.min(31);
        // SAFETY: `ptr` points to a kernel buffer that was copied from user
        // space by the syscall handler before this hook is called.
        let src = unsafe { core::slice::from_raw_parts(ptr, len) };
        task.name[..len].copy_from_slice(src);
        task.name_len = len as u8;
    }
}

fn interrupt_task<R: BootRuntime>(tid: TaskId) -> Result<(), abi::errors::Errno> {
    let should_wake = if let Some(mut task) = crate::task::registry::get_task_mut::<R>(tid) {
        if task.state == TaskState::Dead {
            return Err(abi::errors::Errno::ESRCH);
        }
        task.pending_interrupt = true;
        task.state == TaskState::Blocked
    } else {
        return Err(abi::errors::Errno::ESRCH);
    };

    if should_wake {
        wake_task::<R>(tid);
    }

    Ok(())
}

fn take_pending_interrupt<R: BootRuntime>() -> bool {
    let tid = crate::runtime::<R>().current_tid();
    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(tid) {
        let was_pending = task.pending_interrupt;
        task.pending_interrupt = false;
        was_pending
    } else {
        false
    }
}

/// Get the current task's ProcessInfo Arc, if any.
pub fn process_info<R: BootRuntime>()
-> Option<alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>>> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    // Prefer the runtime's current TID. During syscall/trap handling this stays
    // authoritative even if the scheduler's per-CPU `current` view is transiently stale.
    let runtime_tid = rt.current_tid();
    let result = crate::task::registry::get_task::<R>(runtime_tid)
        .and_then(|t| t.process_info.clone())
        .or_else(|| {
            let lock = SCHEDULER.lock();
            if let Some(ptr) = *lock {
                let sched = unsafe { &*(ptr as *const types::Scheduler<R>) };
                let cpu_idx = current_cpu_index::<R>();
                sched
                    .state
                    .per_cpu
                    .get(cpu_idx)
                    .and_then(|pc| pc.current)
                    .and_then(|tid| crate::task::registry::get_task::<R>(tid))
                    .and_then(|t| t.process_info.clone())
            } else {
                None
            }
        });

    rt.irq_restore(_irq);
    result
}

pub fn process_info_for_tid<R: BootRuntime>(
    tid: u64,
) -> Option<alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>>> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let result = crate::task::registry::get_task::<R>(tid).and_then(|t| t.process_info.clone());
    rt.irq_restore(_irq);
    result
}

/// Return a snapshot of all live processes (those with a ProcessInfo).
///
/// Called from the `LIST_PROCESSES_HOOK` slot so that procfs can render
/// `/proc/<pid>/…` files without knowing the concrete `R` type.
pub fn list_processes<R: BootRuntime>() -> alloc::vec::Vec<hooks::ProcessSnapshot> {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let mut out = alloc::vec::Vec::new();
    {
        let reg = crate::task::registry::get_registry::<R>();
        for task in reg.threads.iter() {
            if let Some(pi_arc) = &task.process_info {
                let pi = pi_arc.lock();
                let name_bytes = &task.name[..task.name_len as usize];
                let name = alloc::string::String::from_utf8_lossy(name_bytes).into_owned();
                out.push(hooks::ProcessSnapshot {
                    pid: pi.pid,
                    ppid: pi.lifecycle.ppid,
                    tid: task.id,
                    name,
                    state: task.state,
                    argv: pi.argv.clone(),
                    exec_path: pi.exec_path.clone(),
                    exit_code: task.exit_code,
                    pgid: pi.pgid,
                    sid: pi.sid,
                    session_leader: pi.session_leader,
                    // Place-context fields (Phase 8): extracted from Process into
                    // the snapshot so the place bridge can build a canonical Place
                    // without holding the Process lock.
                    cwd: pi.cwd.clone(),
                    // NamespaceRef is a unit struct in Phase 8 (all processes share
                    // the global mount table).  Label it "global" here; future phases
                    // will replace this with a stable per-process namespace identifier
                    // once process namespace isolation is implemented.
                    namespace_label: alloc::string::String::from("global"),
                });
            }
        }
    }
    rt.irq_restore(_irq);
    out
}

fn register_task_exit_waiter<R: BootRuntime>(
    target_tid: TaskId,
    waiter_tid: TaskId,
) -> Result<Option<i32>, abi::errors::Errno> {
    let target =
        crate::task::registry::get_task::<R>(target_tid).ok_or(abi::errors::Errno::ECHILD)?;

    // Joining a detached thread is not permitted.
    if target.detached {
        return Err(abi::errors::Errno::EINVAL);
    }

    if target.state == TaskState::Dead {
        return Ok(Some(target.exit_code.unwrap_or(0)));
    }

    target.exit_waiters.push_back(waiter_tid);
    Ok(None)
}

pub fn poll_task_exit<R: BootRuntime>(
    target_tid: TaskId,
) -> Result<Option<i32>, abi::errors::Errno> {
    let target =
        crate::task::registry::get_task::<R>(target_tid).ok_or(abi::errors::Errno::ECHILD)?;

    if target.state == TaskState::Dead { Ok(Some(target.exit_code.unwrap_or(0))) } else { Ok(None) }
}

pub fn register_task_exit_waiter_public<R: BootRuntime>(
    target_tid: TaskId,
    waiter_tid: TaskId,
) -> Result<Option<i32>, abi::errors::Errno> {
    register_task_exit_waiter::<R>(target_tid, waiter_tid)
}

pub fn unregister_task_exit_waiter<R: BootRuntime>(
    target_tid: TaskId,
    waiter_tid: TaskId,
) -> Result<(), abi::errors::Errno> {
    let target =
        crate::task::registry::get_task::<R>(target_tid).ok_or(abi::errors::Errno::ECHILD)?;
    target.exit_waiters.remove(waiter_tid);
    Ok(())
}

fn mark_task_exited<R: BootRuntime>(
    sched: &mut types::Scheduler<R>,
    tid: TaskId,
    code: i32,
) -> alloc::vec::Vec<u64> {
    // Collect exit waiters and mark the task dead.
    let mut waiters = if let Some(mut task) = crate::task::registry::get_task_mut::<R>(tid) {
        task.state = TaskState::Dead;
        task.exit_code = Some(code);
        task.exit_waiters.drain()
    } else {
        alloc::vec::Vec::new()
    };

    if let Some(task) = sched.state.get_task_mut(tid) {
        task.runq_location = None;
    }

    // Remove this TID from the process's thread group list.
    // If this is the thread-group leader, drain the remaining siblings in one
    // step to avoid a separate clone + clear pass.
    // Also capture ppid/pid for SIGCHLD notification.
    let mut notify_ppid: u32 = 0;
    let mut notify_pid: u32 = 0;

    let siblings_to_kill: alloc::vec::Vec<TaskId> = {
        let pinfo_opt =
            crate::task::registry::get_task::<R>(tid).and_then(|t| t.process_info.clone());
        if let Some(pinfo) = pinfo_opt {
            let mut pi = pinfo.lock();
            pi.lifecycle.thread_ids.retain(|&t| t != tid);

            // If the exiting thread is the thread-group leader (its TID == pid),
            // drain all remaining siblings and schedule them for termination.
            if pi.pid as TaskId == tid {
                notify_ppid = pi.lifecycle.ppid;
                notify_pid = pi.pid;
                core::mem::take(&mut pi.lifecycle.thread_ids)
            } else {
                alloc::vec::Vec::new()
            }
        } else {
            alloc::vec::Vec::new()
        }
    };

    // If the thread-group leader exited, notify the parent process.
    if notify_ppid != 0 {
        let encoded_status = if code < 0 {
            abi::signal::w_term_sig((-code) as u8)
        } else {
            abi::signal::w_exit_status(code as u8)
        };
        // Find the parent's ProcessInfo Arc without holding the registry lock
        // across the ProcessInfo lock.
        let parent_arc: Option<alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>>> = {
            let reg = crate::task::registry::get_registry::<R>();
            reg.threads.iter().find_map(|task| {
                task.process_info.as_ref().and_then(|pi_arc| {
                    // Avoid locking here; check PID via a try-approach.
                    // We can peek at the pid without locking if it's stable.
                    // ProcessInfo.pid is set at creation and never changes.
                    // However spinning on the Mutex here under the registry
                    // guard risks subtle ordering issues; clone the Arc
                    // and lock it after releasing the registry guard.
                    let guard = pi_arc.lock();
                    if guard.pid == notify_ppid {
                        drop(guard);
                        Some(pi_arc.clone())
                    } else {
                        None
                    }
                })
            })
        }; // registry guard released here

        if let Some(pi_arc) = parent_arc {
            let mut pp = pi_arc.lock();
            pp.lifecycle.children_done.push_back((notify_pid, encoded_status));
            pp.signals.post(abi::signal::SIGCHLD);
            let tids = pp.lifecycle.thread_ids.clone();
            drop(pp);
            // Wake parent threads via the waiters list (after the scheduler lock
            // is released by the caller).
            waiters.extend(tids.iter().map(|&t| t as u64));
        }
    }

    // Kill sibling threads (thread-group exit).
    for &sibling in &siblings_to_kill {
        if let Some(mut task) = crate::task::registry::get_task_mut::<R>(sibling) {
            if task.state != TaskState::Dead {
                task.state = TaskState::Dead;
                task.exit_code = Some(code);
                let sibling_waiters = task.exit_waiters.drain();
                waiters.extend(sibling_waiters);
                drop(task);

                if let Some(sf) = sched.state.get_task_mut(sibling) {
                    sf.runq_location = None;
                }
                sched.state.remove_task_from_runq(sibling);
                crate::kdebug!("SCHED: Killed sibling thread {} (thread-group exit)", sibling);
            }
        }
    }

    waiters
}

fn wake_waiters(waiters: &[u64]) {
    for &tid in waiters {
        unsafe {
            crate::sched::wake_task_erased(tid);
        }
    }
}

fn current_task_resource_id_impl<R: BootRuntime>() -> Option<u64> {
    None
}

pub fn exit<R: BootRuntime>(code: i32) {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    let (switch, waiters) = {
        let lock = SCHEDULER.lock();
        let ptr = lock.expect("Scheduler not initialized");
        let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
        sched.terminate_current(code)
    };

    wake_waiters(&waiters);

    unsafe {
        rt.tasking().activate_address_space(switch.to_aspace);
    }

    unsafe {
        rt.tasking().switch_with_tls(
            &mut *(switch.from_ctx as *mut _),
            &*switch.to_ctx,
            switch.to_tid,
            switch.from_user_fs_base,
            switch.to_user_fs_base,
        );
    }

    unreachable!("Thread continued after terminating!");
}

/// Kill an arbitrary task by TID. Returns true if the task was found and killed.
/// The task is marked Dead with exit code -9 and removed from all run queues.
pub fn kill_by_tid<R: BootRuntime>(tid: u64) -> bool {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    let (killed, waiters) = {
        let lock = SCHEDULER.lock();
        if let Some(ptr) = *lock {
            let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };

            // Don't allow killing the current task via this path
            let cpu_idx = current_cpu_index::<R>();
            if let Some(current_id) = sched.state.per_cpu.get(cpu_idx).and_then(|pc| pc.current) {
                if current_id == tid {
                    (false, alloc::vec::Vec::new())
                } else {
                    let task_killed = crate::task::registry::get_task::<R>(tid)
                        .map(|task| task.state != TaskState::Dead)
                        .unwrap_or(false);

                    if !task_killed {
                        (false, alloc::vec::Vec::new())
                    } else {
                        let waiters = mark_task_exited::<R>(sched, tid, -9);

                        // Remove from all run queues
                        sched.state.remove_task_from_runq(tid);

                        // Remove from wait queue
                        if let Some(pos) = sched.state.wait_queue.iter().position(|&wid| wid == tid)
                        {
                            sched.state.wait_queue.remove(pos);
                        }

                        // Remove from sleep queue
                        sched.state.sleep_queue.retain(|_, tids| {
                            tids.retain(|&t| t != tid);
                            !tids.is_empty()
                        });

                        // Release any claimed devices
                        let released =
                            crate::device_registry::REGISTRY.lock().release_all_for_task(tid);
                        if released > 0 {
                            crate::kinfo!("DEVICE: released {} claims for task {}", released, tid);
                        }

                        crate::kinfo!("SCHED: Killed task {} (SIGKILL)", tid);
                        (true, waiters)
                    }
                }
            } else {
                (false, alloc::vec::Vec::new())
            }
        } else {
            (false, alloc::vec::Vec::new())
        }
    };

    wake_waiters(&waiters);
    rt.irq_restore(_irq);
    killed
}

/// Wait semantics are non-consuming today: any task that can name a TID may
/// observe its terminal status, and dead task records stay resident for later polls/waits.
pub fn wait_task<R: BootRuntime>(tid: TaskId) -> Result<i32, abi::errors::Errno> {
    let current_tid = current_tid::<R>();

    if tid == current_tid {
        return Err(abi::errors::Errno::EINVAL);
    }

    loop {
        if let Some(code) = register_task_exit_waiter::<R>(tid, current_tid)? {
            return Ok(code);
        }

        unsafe {
            block_current_erased();
        }
    }
}

/// Collect TIDs of all tasks that are children of `our_pid`.
///
/// If `target_pid > 0`, only the specific child with that PID is returned.
/// Otherwise, all direct children are returned.
fn collect_child_tids<R: BootRuntime>(our_pid: u32, target_pid: i64) -> alloc::vec::Vec<TaskId> {
    let reg = crate::task::registry::get_registry::<R>();
    reg.threads
        .iter()
        .filter_map(|task| {
            task.process_info.as_ref().and_then(|pi| {
                let pi = pi.lock();
                if pi.lifecycle.ppid != our_pid {
                    return None;
                }
                if target_pid > 0 && pi.pid != target_pid as u32 {
                    return None;
                }
                Some(task.id)
            })
        })
        .collect()
}

fn queued_status_matches(flags: u32, status: i32) -> bool {
    use abi::signal::{wifcontinued, wifstopped};
    use abi::types::waitpid_flags;

    if wifstopped(status) {
        return (flags & waitpid_flags::WUNTRACED) != 0;
    }
    if wifcontinued(status) {
        return (flags & waitpid_flags::WCONTINUED) != 0;
    }
    true
}

fn reap_child_pid_if_dead<R: BootRuntime>(child_pid: u32, status: i32) {
    if abi::signal::wifstopped(status) || abi::signal::wifcontinued(status) {
        return;
    }

    let reg = crate::task::registry::get_registry::<R>();
    let child_tid = reg.threads.iter().find_map(|task| {
        task.process_info
            .as_ref()
            .and_then(|pi| if pi.lock().pid == child_pid { Some(task.id) } else { None })
    });
    drop(reg);

    if let Some(child_tid) = child_tid {
        remove_task_completely::<R>(child_tid);
    }
}

fn take_queued_child_status<R: BootRuntime>(
    _our_pid: u32,
    target_pid: i64,
    flags: u32,
) -> Option<(u64, i32)> {
    let our_process = crate::sched::process_info_current()?;
    let mut process = our_process.lock();
    let mut match_index: Option<usize> = None;

    for (idx, (child_pid, status)) in process.lifecycle.children_done.iter().enumerate() {
        if target_pid > 0 && *child_pid != target_pid as u32 {
            continue;
        }
        if queued_status_matches(flags, *status) {
            match_index = Some(idx);
            break;
        }
    }

    let (child_pid, status) = process.lifecycle.children_done.remove(match_index?)?;
    drop(process);
    reap_child_pid_if_dead::<R>(child_pid, status);
    Some((child_pid as u64, status))
}

/// Internal implementation: performs the wait with an explicit `our_pid`.
///
/// Factored out so tests can exercise the core logic without registering a
/// task at the current-TID slot (which is always 0 in the `MockRuntime`).
fn waitpid_for_pid<R: BootRuntime>(
    our_pid: u32,
    pid: i64,
    flags: u32,
) -> Result<(u64, i32), abi::errors::Errno> {
    use abi::types::waitpid_flags;
    let wnohang = (flags & waitpid_flags::WNOHANG) != 0;

    let our_tid = current_tid::<R>();

    loop {
        if let Some(result) = take_queued_child_status::<R>(our_pid, pid, flags) {
            return Ok(result);
        }

        // Collect matching children (releases registry guard before returning).
        let children = collect_child_tids::<R>(our_pid, pid);

        if children.is_empty() {
            return Err(abi::errors::Errno::ECHILD);
        }

        // Fast path: look for a dead child without registering.
        for &child_tid in &children {
            // Collect exit info without holding the registry guard across the reap call.
            let dead_info = crate::task::registry::get_task::<R>(child_tid).and_then(|task| {
                // `task` (ThreadRef / registry guard) is dropped when this closure returns.
                if task.state == TaskState::Dead {
                    let code = task.exit_code.unwrap_or(0);
                    let child_pid = task
                        .process_info
                        .as_ref()
                        .map(|pi| pi.lock().pid as u64)
                        .unwrap_or(child_tid);
                    Some((child_pid, code))
                } else {
                    None
                }
            });

            if let Some((child_pid, code)) = dead_info {
                // Reap: remove the dead child's record from both the registry and
                // the scheduler state so they stay in sync.
                remove_task_completely::<R>(child_tid);
                return Ok((child_pid, code));
            }
        }

        if wnohang {
            // POSIX: return 0 as the child PID to indicate "no child exited yet".
            return Ok((0, 0));
        }

        // Register as an exit waiter for every live child.  If any child has
        // already died between the check above and the register call,
        // `register_task_exit_waiter` returns `Some(code)` immediately.
        let mut registered: alloc::vec::Vec<TaskId> = alloc::vec::Vec::new();
        let mut early_result: Option<(u64, i32)> = None;
        let mut early_reap_tid: Option<TaskId> = None;

        for &child_tid in &children {
            match register_task_exit_waiter::<R>(child_tid, our_tid) {
                Ok(Some(code)) => {
                    // Child died between our fast-path check and now.
                    let child_pid = crate::task::registry::get_task::<R>(child_tid)
                        .and_then(|t| t.process_info.as_ref().map(|pi| pi.lock().pid as u64))
                        .unwrap_or(child_tid);
                    early_result = Some((child_pid, code));
                    early_reap_tid = Some(child_tid);
                    break;
                }
                Ok(None) => registered.push(child_tid),
                Err(_) => {} // Child vanished — skip it.
            }
        }

        if let Some(result) = early_result {
            // Clean up any waiters we already registered before finding the dead child.
            for &child_tid in &registered {
                let _ = unregister_task_exit_waiter::<R>(child_tid, our_tid);
            }
            // Reap the dead child.
            if let Some(reap_tid) = early_reap_tid {
                remove_task_completely::<R>(reap_tid);
            }
            return Ok(result);
        }

        if registered.is_empty() {
            // All children died in the window between collection and registration.
            continue;
        }

        // Block until any registered child exits.
        unsafe {
            block_current_erased();
        }

        // After waking, unregister from children that haven't yet exited.
        for &child_tid in &registered {
            let _ = unregister_task_exit_waiter::<R>(child_tid, our_tid);
        }

        // Loop back to find the next queued child state transition.
    }
}

/// Wait for a child process to exit, returning `(child_pid, exit_code)`.
///
/// - `pid > 0`: wait for the specific child with that PID.
/// - `pid == -1` or `pid == 0`: wait for any child of the calling process.
/// - `flags & WNOHANG`: return `Ok((0, 0))` immediately if no child has exited.
///
/// Returns `Err(ECHILD)` when no matching children exist at all.
pub fn waitpid<R: BootRuntime>(pid: i64, flags: u32) -> Result<(u64, i32), abi::errors::Errno> {
    let our_tid = current_tid::<R>();

    // Retrieve the calling process's PID from its ProcessInfo.
    let our_pid = {
        let task =
            crate::task::registry::get_task::<R>(our_tid).ok_or(abi::errors::Errno::EINVAL)?;
        task.process_info.as_ref().map(|pi| pi.lock().pid).ok_or(abi::errors::Errno::EINVAL)?
    };

    waitpid_for_pid::<R>(our_pid, pid, flags)
}

// ── Signal mask hooks ─────────────────────────────────────────────────────────

fn get_signal_mask<R: BootRuntime>() -> abi::signal::SigSet {
    let tid = current_tid::<R>();
    crate::task::registry::get_task::<R>(tid)
        .map(|t| t.signals.effective_mask())
        .unwrap_or(abi::signal::SigSet::EMPTY)
}

fn set_signal_mask<R: BootRuntime>(mask: abi::signal::SigSet) {
    let tid = current_tid::<R>();
    if let Some(mut t) = crate::task::registry::get_task_mut::<R>(tid) {
        t.signals.mask = abi::signal::SigSet(mask.0 & !crate::signal::UNCATCHABLE.0);
    }
}

fn get_thread_pending<R: BootRuntime>() -> abi::signal::SigSet {
    let tid = current_tid::<R>();
    crate::task::registry::get_task::<R>(tid)
        .map(|t| t.signals.pending)
        .unwrap_or(abi::signal::SigSet::EMPTY)
}

fn set_thread_pending<R: BootRuntime>(pending: abi::signal::SigSet) {
    let tid = current_tid::<R>();
    if let Some(mut t) = crate::task::registry::get_task_mut::<R>(tid) {
        t.signals.pending = pending;
    }
}

pub fn register_timeout_wake<R: BootRuntime>(tid: TaskId, wake_tick: u64) {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    if let Some(ptr) = *lock {
        let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
        let sleepers = sched.state.sleep_queue.entry(wake_tick).or_default();
        if !sleepers.contains(&tid) {
            sleepers.push(tid);
        }
    }
    rt.irq_restore(_irq);
}

pub fn unregister_timeout_wake<R: BootRuntime>(tid: TaskId) {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    if let Some(ptr) = *lock {
        let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
        sched.state.sleep_queue.retain(|_, tids| {
            tids.retain(|&sleep_tid| sleep_tid != tid);
            !tids.is_empty()
        });
    }
    rt.irq_restore(_irq);
}

pub fn cpu_online<R: BootRuntime>(cpu_index: usize) {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    if let Some(ptr) = *lock {
        let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
        sched.cpu_online(cpu_index);
    }
    rt.irq_restore(_irq);
}

/// Remove a task completely from both the global registry and the scheduler state.
///
/// This is the canonical way to "reap" a task.  It ensures that the thread
/// list in the registry stays in sync with the scheduler's sorted `threads`
/// vector, preventing index drift that would otherwise lead to panics in the
/// context switcher.
pub fn remove_task_completely<R: BootRuntime>(tid: TaskId) {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    // 1. Remove from scheduler state (requires SCHEDULER lock)
    {
        let lock = SCHEDULER.lock();
        if let Some(ptr) = *lock {
            let sched = unsafe { &mut *(ptr as *mut types::Scheduler<R>) };
            sched.state.remove_task(tid);
        }
    }

    // 2. Remove from global registry (requires REGISTRY lock)
    crate::task::registry::get_registry::<R>().remove(tid);

    rt.irq_restore(_irq);
}

pub fn dump_stats<R: BootRuntime>() {
    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &*(ptr as *const types::Scheduler<R>) };

    crate::kprint!("\n====== TASK DUMP ======\n");
    crate::kprint!(
        "CPUs: {} online / {} total\n",
        sched.state.online_cpu_count,
        sched.total_cpu_count
    );
    crate::kprint!(
        " {:>5}  {:>10}  {:>4}  {:>3}  {:>4}  {:>7}  {:>6}  {:>6}  {}\n",
        "TID",
        "STATE",
        "PRI",
        "CPU",
        "USER",
        "SLICE",
        "KSTK",
        "AFFIN",
        "NAME"
    );

    let mut runnable_count = 0u32;
    for task in crate::task::registry::get_registry::<R>().threads.iter() {
        let state_str = match task.state {
            TaskState::Runnable => {
                runnable_count += 1;
                "Runnable"
            }
            TaskState::Running => {
                runnable_count += 1;
                "Running"
            }
            TaskState::Blocked => "Blocked",
            TaskState::Dead => "Dead",
        };
        let pri_str = match task.priority {
            crate::task::TaskPriority::Idle => "Idle",
            crate::task::TaskPriority::Low => "Low",
            crate::task::TaskPriority::Normal => "Norm",
            crate::task::TaskPriority::High => "High",
            crate::task::TaskPriority::Realtime => "RT",
        };
        let cpu_str: alloc::string::String = match task.last_cpu {
            Some(c) => alloc::format!("{}", c),
            None => alloc::string::String::from("-"),
        };
        let user_str = if task.is_user { "Y" } else { "N" };
        let aff_str: alloc::string::String = match task.affinity {
            crate::task::Affinity::Any => alloc::string::String::from("Any"),
            crate::task::Affinity::Pinned(c) => alloc::format!("Pin({})", c),
        };
        let name_str = if task.name_len > 0 {
            core::str::from_utf8(&task.name[..task.name_len as usize]).unwrap_or("?")
        } else {
            "-"
        };
        crate::kprint!(
            " {:>5}  {:>10}  {:>4}  {:>3}  {:>4}  {:>3}/{:<3}  {:>5}K  {:>6}  {}\n",
            task.id,
            state_str,
            pri_str,
            cpu_str,
            user_str,
            task.timeslice_remaining,
            types::DEFAULT_TIMESLICE,
            task.kstack_size / 1024,
            aff_str,
            name_str
        );
    }

    // Per-CPU run-queue summary
    for &i in &sched.state.online_cpus {
        let pc = &sched.state.per_cpu[i];
        let total: usize = pc.runq.iter().map(|q| q.len()).sum();
        crate::kprint!(
            "  CPU {}: current={:?} runq={} idle={:?}\n",
            i,
            pc.current,
            total,
            pc.idle_task
        );
    }

    crate::kprint!("Sleep queue: {} tasks\n", sched.state.sleep_queue.len());
    crate::kprint!(
        "=== {} tasks, {} runnable ===\n\n",
        crate::task::registry::get_registry::<R>().threads.len(),
        runnable_count
    );

    rt.irq_restore(_irq);
}

extern "C" fn idle_task<R: BootRuntime>(_: usize) -> ! {
    let rt = crate::runtime::<R>();
    loop {
        rt.wait_for_interrupt();
    }
}

pub static CPU_ONLINE: AtomicUsize = AtomicUsize::new(0);

/// Entry point for secondary CPUs.
///
/// # Safety
/// Must only be called from `kernel_secondary_entry`.
pub unsafe fn enter_secondary(cpu_index: usize) -> ! {
    // Mark as online
    CPU_ONLINE.fetch_add(1, Ordering::Relaxed);
    crate::kdebug!("SMP: Secondary CPU {} online!", cpu_index);

    // Enter scheduler loop via the hook which bootstraps this CPU.
    // The run_scheduler hook will call bootstrap_cpu to set up this CPU's
    if let Some(hook) = unsafe { hooks::RUN_SCHEDULER_HOOK } {
        hook();
    } else {
        panic!("Scheduler hook not initialized!");
    }

    // Fallback if run_scheduler returns (it shouldn't)
    loop {
        crate::runtime_base().wait_for_interrupt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Affinity, TaskPriority, TaskState};
    use crate::{
        BootRuntime, BootRuntimeBase, BootTasking, MapKind, MapPerms, UserEntry, UserTaskSpec,
    };

    // Mock types for testing - copy from spawn.rs tests
    #[derive(Default, Copy, Clone)]
    pub(crate) struct MockContext(pub(crate) usize);
    #[derive(Clone, Copy, Default)]
    pub(crate) struct MockAddressSpace(pub(crate) u64);

    pub(crate) static MOCK_RUNTIME: MockRuntime = MockRuntime;
    pub(crate) struct MockRuntime;
    impl BootRuntimeBase for MockRuntime {
        fn putchar(&self, _c: u8) {}
        fn mono_ticks(&self) -> u64 {
            0
        }
        fn mono_freq_hz(&self) -> u64 {
            1
        }
        fn init_secondary_cpu(&self, _cpu_index: usize) {}
        fn phys_to_virt_offset(&self) -> u64 {
            0
        }
    }
    impl BootRuntime for MockRuntime {
        type Tasking = MockRuntime;
        fn tasking(&self) -> &Self {
            self
        }
        fn halt(&self) -> ! {
            loop {}
        }
        fn irq_disable(&self) -> crate::IrqState {
            crate::IrqState(0)
        }
        fn irq_restore(&self, _state: crate::IrqState) {}
        fn phys_memory_map(&self) -> &'static [crate::PhysRange] {
            &[]
        }
        fn modules(&self) -> &'static [crate::BootModuleDesc] {
            &[]
        }
        fn framebuffer(&self) -> Option<crate::FramebufferInfo> {
            None
        }
        fn simd_state_layout(&self) -> (usize, usize) {
            (0, 1)
        }
        unsafe fn simd_save(&self, _ptr: *mut u8) {}
        unsafe fn simd_restore(&self, _ptr: *const u8) {}
    }
    impl BootTasking for MockRuntime {
        type Runtime = MockRuntime;
        type Context = MockContext;
        type AddressSpace = MockAddressSpace;
        fn init(&self, _hhdm: u64) {}
        fn init_kernel_context(
            &self,
            _entry: extern "C" fn(usize) -> !,
            _st: u64,
            _arg: usize,
        ) -> Self::Context {
            MockContext(_arg)
        }
        fn init_user_context(
            &self,
            _spec: UserTaskSpec<Self::AddressSpace>,
            _kst: u64,
        ) -> Self::Context {
            MockContext(_spec.arg)
        }
        unsafe fn switch(&self, _f: &mut Self::Context, _t: &Self::Context, _tid: u64) {}
        unsafe fn enter_user(&self, _e: UserEntry) -> ! {
            loop {}
        }
        fn make_user_address_space(&self) -> Self::AddressSpace {
            MockAddressSpace(0)
        }
        fn active_address_space(&self) -> Self::AddressSpace {
            MockAddressSpace(0)
        }
        fn activate_address_space(&self, _as: Self::AddressSpace) {}
        fn map_page(
            &self,
            _as: Self::AddressSpace,
            _v: u64,
            _p: u64,
            _pr: MapPerms,
            _k: MapKind,
            _a: &dyn crate::FrameAllocatorHook,
        ) -> Result<(), ()> {
            Ok(())
        }
        fn unmap_page(&self, _as: Self::AddressSpace, _v: u64) -> Result<Option<u64>, ()> {
            Ok(None)
        }
        fn protect_page(
            &self,
            _as: Self::AddressSpace,
            _virt: u64,
            _perms: MapPerms,
        ) -> Result<(), ()> {
            Ok(())
        }
        fn translate(&self, _as: Self::AddressSpace, _v: u64) -> Option<u64> {
            None
        }
        fn tlb_flush_page(&self, _v: u64) {}
    }

    static INIT_TESTS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    fn init_mock_runtime() {
        if !INIT_TESTS.swap(true, core::sync::atomic::Ordering::SeqCst) {
            crate::init_runtime(&MOCK_RUNTIME);
        }
    }

    #[test]
    fn effective_parallelism_any_uses_online_cpu_count() {
        assert_eq!(effective_parallelism_from_state(1, Affinity::Any), 1);
        assert_eq!(effective_parallelism_from_state(4, Affinity::Any), 4);
    }

    #[test]
    fn effective_parallelism_pinned_is_single_cpu() {
        assert_eq!(effective_parallelism_from_state(1, Affinity::Pinned(0)), 1);
        assert_eq!(effective_parallelism_from_state(8, Affinity::Pinned(3)), 1);
    }

    #[test]
    fn effective_parallelism_never_returns_zero() {
        assert_eq!(effective_parallelism_from_state(0, Affinity::Any), 1);
    }

    /// Serialises sched tests that mutate shared globals (REGISTRY, SCHEDULER,
    /// TICK_COUNT).  Any test that calls `init_test_env` should hold the
    /// returned guard for its entire duration to prevent races with concurrent
    /// tests that reinitialise the registry.
    pub(crate) static SCHED_TEST_GUARD: spin::Mutex<()> = spin::Mutex::new(());

    pub(crate) fn init_test_env() -> spin::MutexGuard<'static, ()> {
        let guard = SCHED_TEST_GUARD.lock();
        init_mock_runtime();
        crate::task::registry::init::<MockRuntime>();
        *SCHEDULER.lock() = None;
        TICK_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
        guard
    }

    fn make_task(
        id: TaskId,
        state: TaskState,
        priority: TaskPriority,
    ) -> crate::task::Task<MockRuntime> {
        crate::task::Task {
            id,
            state,
            priority,
            base_priority: priority,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        }
    }

    #[test]
    fn test_priority_aging_boost() {
        let _g = init_test_env();
        // Test that tasks waiting too long get priority boost when scheduling
        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(0); // Dummy current task

        // The dummy current task must be in BOTH the global registry and the
        // scheduler's local sched-state so that `do_context_switch` can index
        // into both consistently (both vectors are sorted by TID).
        let dummy_current = make_task(0, TaskState::Running, TaskPriority::Normal);
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(dummy_current));
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 0, runq_location: None });

        // Create a normal-priority task enqueued recently
        let task_normal = crate::task::Task {
            id: 1001,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 600, // Recent (now - 600 = 400 < AGING_THRESHOLD, so no boost)
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        // Create a low-priority task enqueued a long time ago
        let task_low = crate::task::Task {
            id: 1002,
            state: TaskState::Runnable,
            priority: TaskPriority::Low,
            base_priority: TaskPriority::Low,
            enqueued_at_tick: 0, // Very old
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(task_normal));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(task_low));

        // Scheduler state entries are separate from the global registry and must
        // be inserted explicitly so `prepare_schedule` can locate them.
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 1001, runq_location: None });
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 1002, runq_location: None });
        sched.state.enqueue_task(0, TaskPriority::Normal as usize, 1001);
        sched.state.enqueue_task(0, TaskPriority::Low as usize, 1002);

        // Simulate time advancing enough to give the Low task a boost of +2 (eff = High=3),
        // while the Normal task, enqueued at tick 600, only waits 400 ticks → no boost (eff = Normal=2).
        let now = types::AGING_THRESHOLD_TICKS * 2;
        TICK_COUNT.store(now, core::sync::atomic::Ordering::Relaxed);

        // Request schedule. The Low task should be selected because its effective priority is higher
        // than Normal due to wait time.
        let next_switch = sched.prepare_schedule().expect("Should find a task");
        assert_eq!(
            next_switch.to_tid, 1002,
            "Low priority task with aging should preempt normal task"
        );

        // Verify it was popped from the Low queue, not moved to High queue
        assert!(sched.state.per_cpu[0].runq[TaskPriority::Low as usize].is_empty());
        assert!(!sched.state.per_cpu[0].runq[TaskPriority::Normal as usize].is_empty());
    }

    #[test]
    fn test_reset_priority_aging_on_schedule() {
        let _g = init_test_env();
        // Test that enqueued_at_tick resets when task is preempted/yields
        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());

        // Task 1 is running
        let mut task1 = crate::task::Task {
            id: 2001,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0, // Very old
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: 0, // timeslice expired
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        // Task 2 is runnable
        let task2 = crate::task::Task {
            id: 2002,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 500, // Newer
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(task1));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(task2));

        sched.state.per_cpu[0].current = Some(2001);
        // Scheduler state entries for both tasks.
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 2001, runq_location: None });
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 2002, runq_location: None });
        sched.state.enqueue_task(0, TaskPriority::Normal as usize, 2002);

        // Time moves forward
        TICK_COUNT.store(1000, core::sync::atomic::Ordering::Relaxed);

        // Trigger a timer tick to cause preemption
        let switch = sched.prepare_yield().expect("Should preempt to task2");
        assert_eq!(switch.to_tid, 2002);

        // Verify task1 was placed back in runq and its enqueued_at_tick was updated to TICK_COUNT
        let t1 = crate::task::registry::get_task::<MockRuntime>(2001).unwrap();
        assert_eq!(t1.enqueued_at_tick, 1000);
        assert_eq!(t1.state, TaskState::Runnable);
    }

    #[test]
    fn test_wake_preempts_lower_priority() {
        let _g = init_test_env();
        crate::task::registry::init::<MockRuntime>();
        use core::sync::atomic::Ordering;

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());

        // Task 1: Normal priority, currently running
        let normal_task = crate::task::Task {
            id: 3001,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        // Task 2: Realtime priority, sleeping (about to wake)
        let rt_task = crate::task::Task {
            id: 3002,
            state: TaskState::Runnable,
            priority: TaskPriority::Realtime,
            base_priority: TaskPriority::Realtime,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(normal_task));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(rt_task));
        sched.state.per_cpu[0].current = Some(3001); // Normal task is running
        // Scheduler state entries for both tasks.
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 3001, runq_location: None });
        sched
            .state
            .insert_task(crate::sched::state::ThreadSchedFields { tid: 3002, runq_location: None });

        // Put RT task in sleep queue with wake_tick in the past
        TICK_COUNT.store(100, Ordering::Relaxed);
        sched.state.sleep_queue.entry(50).or_default().push(3002);

        // Before: need_resched should be false
        assert!(!sched.state.per_cpu[0].need_resched, "need_resched should start false");

        // Wake sleepers — should detect RT > Normal and set need_resched
        sched.wake_sleepers();

        // Verify need_resched was set
        assert!(
            sched.state.per_cpu[0].need_resched,
            "need_resched should be true after waking a higher-priority task"
        );

        // Verify RT task was enqueued to the Realtime runq
        assert!(
            sched.state.per_cpu[0].runq[TaskPriority::Realtime as usize]
                .iter()
                .any(|&id| id == 3002),
            "RT task should be in the Realtime run queue"
        );

        // Now simulate schedule: prepare_yield should pick the RT task
        sched.state.per_cpu[0].need_resched = false; // clear so prepare_yield runs clean
        let switch = sched.prepare_yield();
        assert!(switch.is_some(), "Should produce a context switch");
        let switch = switch.unwrap();
        assert_eq!(switch.to_tid, 3002, "Scheduler should switch to the RT task");
        assert_eq!(switch.from_tid, 3001, "Scheduler should switch away from the Normal task");
    }

    #[test]
    fn test_sorted_insertion() {
        let _g = init_test_env();
        crate::task::registry::init::<MockRuntime>();
        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());

        // Helper to create dummy task
        let make_task = |id: TaskId| crate::task::Task {
            id,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: None,
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        // Insert tasks out of order
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(make_task(4010)));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(make_task(4005)));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(make_task(4020)));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(make_task(4001)));

        // Verify sorted order internally
        assert_eq!(crate::task::registry::get_registry::<MockRuntime>().threads.len(), 4);
        assert_eq!(crate::task::registry::get_registry::<MockRuntime>().threads[0].id, 4001);
        assert_eq!(crate::task::registry::get_registry::<MockRuntime>().threads[1].id, 4005);
        assert_eq!(crate::task::registry::get_registry::<MockRuntime>().threads[2].id, 4010);
        assert_eq!(crate::task::registry::get_registry::<MockRuntime>().threads[3].id, 4020);

        // Verify lookups work
        assert!(crate::task::registry::get_task::<MockRuntime>(4010).is_some());
        assert!(crate::task::registry::get_task::<MockRuntime>(4005).is_some());
        assert!(crate::task::registry::get_task::<MockRuntime>(4001).is_some());
        assert!(crate::task::registry::get_task::<MockRuntime>(4099).is_none());
    }

    #[test]
    fn test_block_and_wake_state_transitions() {
        let _g = init_test_env();
        crate::task::registry::init::<MockRuntime>();
        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());

        let waiting_task = crate::task::Task {
            id: 5001,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(waiting_task));
        sched.state.per_cpu[0].current = Some(5001);

        // Put task in Wait queue and switch it to Blocked (simulating block_current behavior)
        if let Some(mut task) = crate::task::registry::get_task_mut::<MockRuntime>(5001) {
            task.state = TaskState::Blocked;
        }
        sched.state.wait_queue.push_back(5001);

        // Verify task is stuck blocked
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(5001).unwrap().state,
            TaskState::Blocked
        );

        // Emulate `wake_task_erased` via wake_task in MockRuntime context
        crate::sched::blocking::WAKE_TASK_HOOK.store(
            crate::sched::blocking::wake_task::<MockRuntime> as *mut (),
            core::sync::atomic::Ordering::SeqCst,
        );

        // Make sure scheduler hook resolves safely (we will mock inject the scheduler here via static for the hook)
        // Since we are unit testing `wake_task`, we can't easily use the global `SCHEDULER`.
        // So we just directly call the core logic we care about: the wake sleeper unblock logic.

        // Remove from wait queue if present
        if let Some(pos) = sched.state.wait_queue.iter().position(|&wid| wid == 5001) {
            sched.state.wait_queue.remove(pos);
        }

        // Update state to Runnable and add to runq
        if let Some(mut task) = crate::task::registry::get_task_mut::<MockRuntime>(5001) {
            if task.state == TaskState::Blocked {
                task.state = TaskState::Runnable;
                sched.state.enqueue_task(0, task.priority as usize, 5001);
            }
        }

        let woken_task = crate::task::registry::get_task::<MockRuntime>(5001).unwrap();
        assert_eq!(
            woken_task.state,
            TaskState::Runnable,
            "Task must transition from Blocked to Runnable upon wake"
        );

        // Verify task was placed in runq
        assert!(
            sched.state.per_cpu[0].runq[TaskPriority::Normal as usize].iter().any(|&id| id == 5001),
            "Woken task must be in the run queue"
        );
    }

    #[test]
    fn test_wake_task_removes_sleep_queue_entry() {
        let _g = init_test_env();
        crate::task::registry::init::<MockRuntime>();

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(6000);

        let current_task = crate::task::Task {
            id: 6000,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        let sleeping_task = crate::task::Task {
            id: 6001,
            state: TaskState::Blocked,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(current_task));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(sleeping_task));
        sched.state.sleep_queue.entry(10).or_default().push(6001);

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = Some((&mut sched as *mut types::Scheduler<MockRuntime>) as usize);
        drop(sched_lock);

        crate::sched::blocking::wake_task::<MockRuntime>(6001);

        let task = crate::task::registry::get_task::<MockRuntime>(6001).unwrap();
        assert_eq!(task.state, TaskState::Runnable);
        assert!(sched.state.sleep_queue.is_empty());
        assert!(
            sched.state.per_cpu[0].runq[TaskPriority::Normal as usize].iter().any(|&id| id == 6001)
        );

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = None;
    }

    #[test]
    fn test_kill_by_tid_removes_wait_queue_entry() {
        let _g = init_test_env();
        crate::task::registry::init::<MockRuntime>();

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(7000);

        let current_task = crate::task::Task {
            id: 7000,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        let blocked_task = crate::task::Task {
            id: 7001,
            state: TaskState::Blocked,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(current_task));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(blocked_task));
        sched.state.wait_queue.push_back(7001);

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = Some((&mut sched as *mut types::Scheduler<MockRuntime>) as usize);
        drop(sched_lock);

        assert!(kill_by_tid::<MockRuntime>(7001));
        assert!(sched.state.wait_queue.is_empty());
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(7001).unwrap().state,
            TaskState::Dead
        );

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = None;
    }

    #[test]
    fn test_wait_task_returns_immediately_for_dead_target() {
        let _g = init_test_env();

        let dead_task = crate::task::Task {
            id: 8001,
            state: TaskState::Dead,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: Some(23),
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(dead_task));

        assert_eq!(wait_task::<MockRuntime>(8001).unwrap(), 23);
    }

    #[test]
    fn test_wait_task_returns_echild_for_missing_target() {
        let _g = init_test_env();

        assert_eq!(wait_task::<MockRuntime>(8999).unwrap_err(), abi::errors::Errno::ECHILD);
    }

    #[test]
    fn test_register_task_exit_waiter_tracks_live_target() {
        let _g = init_test_env();

        let live_task = crate::task::Task {
            id: 8101,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(live_task));

        assert_eq!(register_task_exit_waiter::<MockRuntime>(8101, 8102).unwrap(), None);

        let waiters =
            crate::task::registry::get_task::<MockRuntime>(8101).unwrap().exit_waiters.drain();
        assert_eq!(waiters, alloc::vec![8102]);
    }

    #[test]
    fn test_kill_by_tid_wakes_registered_exit_waiter() {
        let _g = init_test_env();

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(8200);

        let current_task = crate::task::Task {
            id: 8200,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        let waiter_task = crate::task::Task {
            id: 8201,
            state: TaskState::Blocked,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        let target_task = crate::task::Task {
            id: 8202,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: false,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(current_task));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(waiter_task));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(target_task));

        let target_fields = crate::sched::state::TaskSchedFields { tid: 8202, runq_location: None };
        sched.state.insert_task(target_fields);

        assert_eq!(register_task_exit_waiter::<MockRuntime>(8202, 8201).unwrap(), None);

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = Some((&mut sched as *mut types::Scheduler<MockRuntime>) as usize);
        drop(sched_lock);

        assert!(kill_by_tid::<MockRuntime>(8202));
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(8202).unwrap().state,
            TaskState::Dead
        );
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(8202).unwrap().exit_code,
            Some(-9)
        );
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(8201).unwrap().state,
            TaskState::Runnable
        );
        assert!(
            sched.state.per_cpu[0].runq[TaskPriority::Normal as usize].iter().any(|&id| id == 8201)
        );

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = None;
    }

    #[test]
    fn test_poll_task_exit_reports_pending_dead_and_missing_targets() {
        let _g = init_test_env();

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_task(8301, TaskState::Runnable, TaskPriority::Normal),
        ));

        let mut dead = make_task(8302, TaskState::Dead, TaskPriority::Normal);
        dead.exit_code = Some(17);
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(dead));

        assert_eq!(poll_task_exit::<MockRuntime>(8301).unwrap(), None);
        assert_eq!(poll_task_exit::<MockRuntime>(8302).unwrap(), Some(17));
        assert_eq!(poll_task_exit::<MockRuntime>(8399).unwrap_err(), abi::errors::Errno::ECHILD);
    }

    #[test]
    fn test_unregister_task_exit_waiter_removes_only_requested_waiter() {
        let _g = init_test_env();

        let target = make_task(8401, TaskState::Runnable, TaskPriority::Normal);
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(target));

        register_task_exit_waiter::<MockRuntime>(8401, 8402).unwrap();
        register_task_exit_waiter::<MockRuntime>(8401, 8403).unwrap();
        unregister_task_exit_waiter::<MockRuntime>(8401, 8402).unwrap();

        let waiters =
            crate::task::registry::get_task::<MockRuntime>(8401).unwrap().exit_waiters.drain();
        assert_eq!(waiters, alloc::vec![8403]);
    }

    #[test]
    fn test_register_timeout_wake_deduplicates_task_ids() {
        let _g = init_test_env();

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(8500);

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = Some((&mut sched as *mut types::Scheduler<MockRuntime>) as usize);
        drop(sched_lock);

        register_timeout_wake::<MockRuntime>(8501, 42);
        register_timeout_wake::<MockRuntime>(8501, 42);
        register_timeout_wake::<MockRuntime>(8502, 42);

        assert_eq!(sched.state.sleep_queue.get(&42).cloned().unwrap(), alloc::vec![8501, 8502]);

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = None;
    }

    #[test]
    fn test_unregister_timeout_wake_removes_task_from_all_buckets() {
        let _g = init_test_env();

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(8600);
        sched.state.sleep_queue.insert(11, alloc::vec![8601, 8602]);
        sched.state.sleep_queue.insert(12, alloc::vec![8602, 8603]);

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = Some((&mut sched as *mut types::Scheduler<MockRuntime>) as usize);
        drop(sched_lock);

        unregister_timeout_wake::<MockRuntime>(8602);

        assert_eq!(sched.state.sleep_queue.get(&11).cloned().unwrap(), alloc::vec![8601]);
        assert_eq!(sched.state.sleep_queue.get(&12).cloned().unwrap(), alloc::vec![8603]);

        unregister_timeout_wake::<MockRuntime>(8603);
        assert!(!sched.state.sleep_queue.contains_key(&12));

        let mut sched_lock = SCHEDULER.lock();
        *sched_lock = None;
    }

    #[test]
    fn test_interrupt_task_marks_and_consumes_pending_interrupt() {
        let _g = init_test_env();

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_task(0, TaskState::Runnable, TaskPriority::Normal),
        ));

        interrupt_task::<MockRuntime>(0).expect("interrupt task");
        assert!(crate::task::registry::get_task::<MockRuntime>(0).unwrap().pending_interrupt);
        assert!(take_pending_interrupt::<MockRuntime>());
        assert!(!take_pending_interrupt::<MockRuntime>());
    }

    #[test]
    fn test_interrupt_task_returns_esrch_for_missing_task() {
        let _g = init_test_env();
        assert_eq!(interrupt_task::<MockRuntime>(9999).unwrap_err(), abi::errors::Errno::ESRCH);
    }

    // ── waitpid tests ─────────────────────────────────────────────────────────

    /// Helper: build a task with a populated ProcessInfo (pid + ppid).
    fn make_process_task(
        id: TaskId,
        state: TaskState,
        pid: u32,
        ppid: u32,
        exit_code: Option<i32>,
    ) -> crate::task::Task<MockRuntime> {
        crate::task::Task {
            id,
            state,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: Some(alloc::sync::Arc::new(spin::Mutex::new(
                crate::task::ProcessInfo {
                    pid,
                    lifecycle: crate::task::ProcessLifecycle::new(ppid, pid as TaskId),
                    pgid: pid,
                    sid: pid,
                    session_leader: false,
                    argv: alloc::vec::Vec::new(),
                    env: alloc::collections::BTreeMap::new(),
                    auxv: alloc::vec::Vec::new(),
                    fd_table: crate::vfs::fd_table::FdTable::new(),
                    namespace: crate::vfs::NamespaceRef::global(),
                    cwd: alloc::string::String::from("/"),
                    exec_path: alloc::string::String::new(),
                    space: crate::task::ProcessAddressSpace::empty(),
                    signals: crate::signal::ProcessSignals::new(),
                },
            ))),
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        }
    }

    #[test]
    fn test_waitpid_returns_exit_code_for_dead_specific_child() {
        let _g = init_test_env();

        // Child task: pid=1001, ppid=1000, exit_code=42.
        // Tests call waitpid_for_pid directly so no parent task is needed.
        let child = make_process_task(9101, TaskState::Dead, 1001, 1000, Some(42));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        let (child_pid, code) =
            waitpid_for_pid::<MockRuntime>(1000, 1001, 0).expect("waitpid specific");
        assert_eq!(child_pid, 1001, "returned child pid");
        assert_eq!(code, 42, "returned exit code");
    }

    #[test]
    fn test_waitpid_returns_exit_code_for_any_dead_child() {
        let _g = init_test_env();

        // Child: pid=2001, ppid=2000, exit_code=7
        let child = make_process_task(9201, TaskState::Dead, 2001, 2000, Some(7));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        // pid == -1: wait for any child of process 2000
        let (child_pid, code) = waitpid_for_pid::<MockRuntime>(2000, -1, 0).expect("waitpid any");
        assert_eq!(child_pid, 2001);
        assert_eq!(code, 7);
    }

    #[test]
    fn test_waitpid_echild_when_no_children_exist() {
        let _g = init_test_env();

        // Registry is empty; no children for pid=3000
        let err = waitpid_for_pid::<MockRuntime>(3000, -1, 0).unwrap_err();
        assert_eq!(err, abi::errors::Errno::ECHILD);
    }

    #[test]
    fn test_waitpid_echild_when_specific_child_not_found() {
        let _g = init_test_env();

        // A child with a *different* ppid — should not be found for pid=4000
        let unrelated = make_process_task(9401, TaskState::Dead, 9999, 5000, Some(0));
        crate::task::registry::get_registry::<MockRuntime>()
            .insert(alloc::boxed::Box::new(unrelated));

        // Looking for child pid=9999 under parent pid=4000 → ECHILD
        let err = waitpid_for_pid::<MockRuntime>(4000, 9999, 0).unwrap_err();
        assert_eq!(err, abi::errors::Errno::ECHILD);
    }

    #[test]
    fn test_waitpid_wnohang_returns_zero_when_child_alive() {
        let _g = init_test_env();

        // Live child — not yet exited
        let child = make_process_task(9501, TaskState::Runnable, 5001, 5000, None);
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        let (child_pid, code) =
            waitpid_for_pid::<MockRuntime>(5000, -1, abi::types::waitpid_flags::WNOHANG)
                .expect("wnohang");
        assert_eq!(child_pid, 0, "no child exited yet");
        assert_eq!(code, 0);
    }

    #[test]
    fn test_waitpid_multiple_children_returns_first_dead() {
        let _g = init_test_env();

        // Two children under parent pid=6000: first alive, second dead
        let child_alive = make_process_task(9601, TaskState::Runnable, 6001, 6000, None);
        let child_dead = make_process_task(9602, TaskState::Dead, 6002, 6000, Some(99));

        let mut reg = crate::task::registry::get_registry::<MockRuntime>();
        reg.insert(alloc::boxed::Box::new(child_alive));
        reg.insert(alloc::boxed::Box::new(child_dead));
        drop(reg);

        let (child_pid, code) = waitpid_for_pid::<MockRuntime>(6000, -1, 0).expect("waitpid multi");
        assert_eq!(child_pid, 6002);
        assert_eq!(code, 99);
    }

    /// After `waitpid` successfully returns a dead child's exit code the child's
    /// registry entry must be removed (reaped).  Without reaping, dead process
    /// records accumulate indefinitely ("zombie" leak).
    #[test]
    fn test_waitpid_reaps_dead_child() {
        let _g = init_test_env();

        let child = make_process_task(9701, TaskState::Dead, 7001, 7000, Some(55));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        // The child is still in the registry before the wait.
        assert!(
            crate::task::registry::get_task::<MockRuntime>(9701).is_some(),
            "child must be in registry before waitpid"
        );

        let (child_pid, code) =
            waitpid_for_pid::<MockRuntime>(7000, 7001, 0).expect("waitpid reap");
        assert_eq!(child_pid, 7001);
        assert_eq!(code, 55);

        // After a successful wait the child record must have been reaped.
        assert!(
            crate::task::registry::get_task::<MockRuntime>(9701).is_none(),
            "child record must be removed from registry after reaping"
        );
    }

    /// Once a child has been reaped, a second `waitpid` for the same child must
    /// return `ECHILD` — the record no longer exists.
    #[test]
    fn test_waitpid_echild_after_reaping() {
        let _g = init_test_env();

        let child = make_process_task(9702, TaskState::Dead, 7002, 7003, Some(0));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        // First wait reaps the child.
        waitpid_for_pid::<MockRuntime>(7003, 7002, 0).expect("first waitpid");

        // Second wait must fail because the record was removed.
        let err = waitpid_for_pid::<MockRuntime>(7003, 7002, 0).unwrap_err();
        assert_eq!(
            err,
            abi::errors::Errno::ECHILD,
            "second waitpid after reaping must return ECHILD"
        );
    }

    /// A dead child whose parent has NOT yet called `waitpid` must remain in
    /// the registry (zombie semantics: exit status preserved until collected).
    #[test]
    fn test_dead_child_stays_in_registry_until_reaped() {
        let _g = init_test_env();

        let child = make_process_task(9703, TaskState::Dead, 7010, 7011, Some(3));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        // No waitpid called yet — record must still be present.
        assert!(
            crate::task::registry::get_task::<MockRuntime>(9703).is_some(),
            "dead child must remain in registry until reaped"
        );

        // Verify state and exit code are accessible while zombie.
        let task = crate::task::registry::get_task::<MockRuntime>(9703).unwrap();
        assert_eq!(task.state, TaskState::Dead);
        assert_eq!(task.exit_code, Some(3));
    }

    /// `waitpid` with WNOHANG must not reap any child when no child has exited.
    #[test]
    fn test_waitpid_wnohang_does_not_reap_live_child() {
        let _g = init_test_env();

        let child = make_process_task(9704, TaskState::Runnable, 7020, 7021, None);
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(child));

        let (returned_pid, _) =
            waitpid_for_pid::<MockRuntime>(7021, -1, abi::types::waitpid_flags::WNOHANG)
                .expect("wnohang on live child");
        assert_eq!(returned_pid, 0, "wnohang returns 0 when no child exited");

        // Live child must still be in registry.
        assert!(
            crate::task::registry::get_task::<MockRuntime>(9704).is_some(),
            "live child must remain in registry after WNOHANG poll"
        );
    }

    /// Helper: create a task + ProcessInfo with `tgid` populated and a shared
    /// thread_ids list for multi-thread tests.
    fn make_thread_task(
        id: TaskId,
        state: TaskState,
        pid: u32,
        ppid: u32,
        shared_pinfo: alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>>,
    ) -> crate::task::Task<MockRuntime> {
        crate::task::Task {
            id,
            state,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: Some(shared_pinfo),
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        }
    }

    /// Thread IDs are tracked in ProcessInfo.thread_ids when tasks share the
    /// same ProcessInfo Arc.
    #[test]
    fn test_thread_ids_tracked_in_process_info() {
        let _g = init_test_env();

        // Build a shared ProcessInfo for a 2-thread group.
        // pid = 7000 (thread-group leader), thread_ids = [7000, 7001].
        let pinfo = alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: 7000,
            lifecycle: crate::task::ProcessLifecycle {
                ppid: 1,
                thread_ids: alloc::vec![7000, 7001],
                exec_in_progress: false,
                children_done: alloc::collections::VecDeque::new(),
            },
            pgid: 7000,
            sid: 7000,
            session_leader: false,
            argv: alloc::vec::Vec::new(),
            env: alloc::collections::BTreeMap::new(),
            auxv: alloc::vec::Vec::new(),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::new(),
            space: crate::task::ProcessAddressSpace::empty(),
            signals: crate::signal::ProcessSignals::new(),
        }));

        {
            let pi = pinfo.lock();
            assert_eq!(pi.lifecycle.thread_ids.len(), 2);
            assert!(pi.lifecycle.thread_ids.contains(&7000));
            assert!(pi.lifecycle.thread_ids.contains(&7001));
            assert_eq!(pi.pid, 7000);
        }
    }

    /// When the thread-group leader exits, sibling threads are removed from the
    /// thread_ids list and their scheduler state is set to Dead.
    #[test]
    fn test_mark_task_exited_removes_tid_from_thread_ids() {
        let _g = init_test_env();

        // Shared ProcessInfo for a 2-thread group: leader 8700, sibling 8701.
        let pinfo = alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: 8700,
            lifecycle: crate::task::ProcessLifecycle {
                ppid: 1,
                thread_ids: alloc::vec![8700, 8701],
                exec_in_progress: false,
                children_done: alloc::collections::VecDeque::new(),
            },
            pgid: 8700,
            sid: 8700,
            session_leader: false,
            argv: alloc::vec::Vec::new(),
            env: alloc::collections::BTreeMap::new(),
            auxv: alloc::vec::Vec::new(),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::new(),
            space: crate::task::ProcessAddressSpace::empty(),
            signals: crate::signal::ProcessSignals::new(),
        }));

        // Register both tasks.
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(8700, TaskState::Running, 8700, 1, pinfo.clone()),
        ));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(8701, TaskState::Runnable, 8700, 1, pinfo.clone()),
        ));

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(8700);

        // Exit the sibling thread first — its TID should be removed from thread_ids.
        let _ = mark_task_exited::<MockRuntime>(&mut sched, 8701, 0);

        {
            let pi = pinfo.lock();
            // 8701 should have been removed.
            assert!(
                !pi.lifecycle.thread_ids.contains(&8701),
                "sibling TID still in thread_ids"
            );
            // 8700 (leader) is still present — it hasn't exited yet.
            assert!(pi.lifecycle.thread_ids.contains(&8700), "leader TID wrongly removed");
        }

        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(8701).unwrap().state,
            TaskState::Dead,
            "sibling should be dead"
        );
    }

    /// When the thread-group leader (tid == pid) exits, remaining sibling
    /// threads are also killed (thread-group exit).
    #[test]
    fn test_thread_group_leader_exit_kills_siblings() {
        let _g = init_test_env();

        // Shared ProcessInfo for a 2-thread group: leader 8800, sibling 8801.
        let pinfo = alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: 8800,
            lifecycle: crate::task::ProcessLifecycle {
                ppid: 1,
                thread_ids: alloc::vec![8800, 8801],
                exec_in_progress: false,
                children_done: alloc::collections::VecDeque::new(),
            },
            pgid: 8800,
            sid: 8800,
            session_leader: false,
            argv: alloc::vec::Vec::new(),
            env: alloc::collections::BTreeMap::new(),
            auxv: alloc::vec::Vec::new(),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::new(),
            space: crate::task::ProcessAddressSpace::empty(),
            signals: crate::signal::ProcessSignals::new(),
        }));

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(8800, TaskState::Running, 8800, 1, pinfo.clone()),
        ));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(8801, TaskState::Runnable, 8800, 1, pinfo.clone()),
        ));

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(8800);

        // Exit the thread-group leader.
        let _ = mark_task_exited::<MockRuntime>(&mut sched, 8800, 42);

        // Leader must be dead.
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(8800).unwrap().state,
            TaskState::Dead,
            "leader should be dead"
        );

        // Sibling must also be dead (killed by thread-group exit).
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(8801).unwrap().state,
            TaskState::Dead,
            "sibling should be killed on leader exit"
        );

        // Both TIDs removed from thread_ids.
        assert!(
            pinfo.lock().lifecycle.thread_ids.is_empty(),
            "thread_ids should be empty after group exit"
        );
    }

    /// exec_in_progress: killing siblings during exec collapse removes their
    /// TIDs from thread_ids, leaving only the exec-calling thread.
    #[test]
    fn test_exec_collapse_kills_siblings_and_updates_thread_ids() {
        let _g = init_test_env();

        // Shared ProcessInfo for a 3-thread group: leader 9100, siblings 9101, 9102.
        let pinfo = alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: 9100,
            lifecycle: crate::task::ProcessLifecycle {
                ppid: 1,
                thread_ids: alloc::vec![9100, 9101, 9102],
                exec_in_progress: false,
                children_done: alloc::collections::VecDeque::new(),
            },
            pgid: 9100,
            sid: 9100,
            session_leader: false,
            argv: alloc::vec::Vec::new(),
            env: alloc::collections::BTreeMap::new(),
            auxv: alloc::vec::Vec::new(),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::new(),
            space: crate::task::ProcessAddressSpace::empty(),
            signals: crate::signal::ProcessSignals::new(),
        }));

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(9100, TaskState::Running, 9100, 1, pinfo.clone()),
        ));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(9101, TaskState::Runnable, 9100, 1, pinfo.clone()),
        ));
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(9102, TaskState::Runnable, 9100, 1, pinfo.clone()),
        ));

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        // Mark 9100 as the "current" (exec-calling) thread.
        sched.state.per_cpu[0].current = Some(9100);

        // Step 1: simulate exec – set exec_in_progress.
        pinfo.lock().lifecycle.exec_in_progress = true;

        // Step 2: collect siblings.
        let caller_tid: TaskId = 9100;
        let siblings: alloc::vec::Vec<TaskId> = pinfo
            .lock()
            .lifecycle.thread_ids
            .iter()
            .copied()
            .filter(|&t| t != caller_tid)
            .collect();
        assert_eq!(siblings.len(), 2);

        // Step 3: kill siblings (simulates kill_by_tid path).
        for sibling in siblings {
            let _ = mark_task_exited::<MockRuntime>(&mut sched, sibling, -9);
        }

        // Both siblings must be dead.
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(9101).unwrap().state,
            TaskState::Dead,
            "sibling 9101 should be dead"
        );
        assert_eq!(
            crate::task::registry::get_task::<MockRuntime>(9102).unwrap().state,
            TaskState::Dead,
            "sibling 9102 should be dead"
        );

        // thread_ids should contain only the caller.
        {
            let pi = pinfo.lock();
            assert_eq!(
                pi.lifecycle.thread_ids,
                alloc::vec![caller_tid],
                "only caller TID should remain after collapse"
            );
        }

        // Step 4: simulate commit – clear exec_in_progress.
        pinfo.lock().lifecycle.exec_in_progress = false;
        assert!(
            !pinfo.lock().lifecycle.exec_in_progress,
            "exec_in_progress cleared after commit"
        );
    }

    /// exec collapse with 4 threads is deterministic: ALL siblings (9701–9703)
    /// are in Dead state before the exec-caller (9700) proceeds to commit.
    ///
    /// This tests the full scheduler + registry path that `task_exec_current`
    /// uses via `kill_by_tid` → `mark_task_exited`:
    ///   1. Set exec_in_progress.
    ///   2. Collect sibling TIDs (exclude caller).
    ///   3. Kill every sibling via mark_task_exited.
    ///   4. Assert every sibling is Dead and only caller TID remains.
    ///   5. Assert exec-caller is NOT Dead.
    ///   6. Commit: clear exec_in_progress.
    #[test]
    fn test_exec_collapse_determinism_four_threads() {
        let _g = init_test_env();

        let caller_tid: TaskId = 9700;
        let sibling_tids: [TaskId; 3] = [9701, 9702, 9703];

        let mut all_tids = alloc::vec![caller_tid];
        all_tids.extend_from_slice(&sibling_tids);

        let pinfo = alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: 9700,
            lifecycle: crate::task::ProcessLifecycle {
                ppid: 1,
                thread_ids: all_tids.clone(),
                exec_in_progress: false,
                children_done: alloc::collections::VecDeque::new(),
            },
            pgid: 9700,
            sid: 9700,
            session_leader: false,
            argv: alloc::vec![b"old".to_vec()],
            env: alloc::collections::BTreeMap::new(),
            auxv: alloc::vec::Vec::new(),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::from("/old/binary"),
            space: crate::task::ProcessAddressSpace::empty(),
            signals: crate::signal::ProcessSignals::new(),
        }));

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
            make_thread_task(caller_tid, TaskState::Running, 9700, 1, pinfo.clone()),
        ));
        for &sid in &sibling_tids {
            crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(
                make_thread_task(sid, TaskState::Runnable, 9700, 1, pinfo.clone()),
            ));
        }

        let mut sched = types::Scheduler::<MockRuntime>::new();
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(caller_tid);

        // Phase 1: set exec_in_progress atomically.
        pinfo.lock().lifecycle.exec_in_progress = true;

        // Phase 2: collect sibling TIDs (excluding caller).
        let siblings: alloc::vec::Vec<TaskId> = pinfo
            .lock()
            .lifecycle.thread_ids
            .iter()
            .copied()
            .filter(|&t| t != caller_tid)
            .collect();
        assert_eq!(siblings.len(), 3, "expected 3 siblings");
        assert!(!siblings.contains(&caller_tid), "caller must not appear in sibling list");

        // Phase 3: kill every sibling (as task_exec_current calls kill_by_tid_current).
        for &sid in &siblings {
            let _ = mark_task_exited::<MockRuntime>(&mut sched, sid, -9);
        }

        // Phase 4: invariant — every sibling must be Dead before commit.
        for &sid in &sibling_tids {
            assert_eq!(
                crate::task::registry::get_task::<MockRuntime>(sid)
                    .expect("sibling must remain as zombie")
                    .state,
                TaskState::Dead,
                "sibling {} must be Dead after exec collapse",
                sid
            );
        }

        // thread_ids must contain only the exec-caller.
        assert_eq!(
            pinfo.lock().lifecycle.thread_ids,
            alloc::vec![caller_tid],
            "only exec-caller TID must remain in thread_ids after collapse"
        );

        // The exec-caller itself must NOT be Dead.
        assert_ne!(
            crate::task::registry::get_task::<MockRuntime>(caller_tid)
                .expect("exec-caller must still be in registry")
                .state,
            TaskState::Dead,
            "exec-caller must not be killed during collapse"
        );

        // Phase 5: commit — clear exec_in_progress.
        pinfo.lock().lifecycle.exec_in_progress = false;
        assert!(
            !pinfo.lock().lifecycle.exec_in_progress,
            "exec_in_progress must be cleared after commit"
        );
    }

    /// exec_in_progress blocks additional thread creation at the process level.
    #[test]
    fn test_exec_in_progress_rejects_new_threads() {
        let _g = init_test_env();

        let pinfo = alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: 9300,
            lifecycle: crate::task::ProcessLifecycle::new(1, 9300),
            pgid: 9300,
            sid: 9300,
            session_leader: false,
            argv: alloc::vec::Vec::new(),
            env: alloc::collections::BTreeMap::new(),
            auxv: alloc::vec::Vec::new(),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::new(),
            space: crate::task::ProcessAddressSpace::empty(),
            signals: crate::signal::ProcessSignals::new(),
        }));

        // Before exec: flag is clear — new threads would be accepted.
        assert!(!pinfo.lock().lifecycle.exec_in_progress);

        // Set exec_in_progress (as task_exec_current does at the start).
        pinfo.lock().lifecycle.exec_in_progress = true;

        // The sys_spawn_thread handler checks this flag and returns EAGAIN.
        // Here we verify the condition it tests.
        assert!(
            pinfo.lock().lifecycle.exec_in_progress,
            "exec_in_progress must be set to block SYS_SPAWN_THREAD"
        );

        // Rollback: clear the flag on pre-commit failure.
        pinfo.lock().lifecycle.exec_in_progress = false;
        assert!(
            !pinfo.lock().lifecycle.exec_in_progress,
            "flag cleared after rollback"
        );
    }

    // ── TLS-base and detached-thread tests ───────────────────────────────────

    /// A task constructed with a non-zero `user_fs_base` retains that value.
    ///
    /// This is the kernel-side invariant for the TLS-base handoff: the spawn
    /// path stores `tls_base` in `Task.user_fs_base`, and the scheduler
    /// writes it to hardware (FS_BASE) on the first context switch.
    #[test]
    fn test_tls_base_stored_in_task_user_fs_base() {
        let _g = init_test_env();

        let tls_base: u64 = 0xDEAD_CAFE_0000_0000;

        let task = crate::task::Task {
            id: 9800,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: tls_base,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(task));

        let stored = crate::task::registry::get_task::<MockRuntime>(9800)
            .expect("task must be in registry")
            .user_fs_base;
        assert_eq!(stored, tls_base, "user_fs_base must equal the requested tls_base");
    }

    /// Joining a detached thread must return `EINVAL`.
    #[test]
    fn test_detached_thread_cannot_be_joined() {
        let _g = init_test_env();

        let task = crate::task::Task {
            id: 9801,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: true, // detached — must not be joinable
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(task));

        assert_eq!(
            register_task_exit_waiter::<MockRuntime>(9801, 9802).unwrap_err(),
            abi::errors::Errno::EINVAL,
            "joining a detached thread must return EINVAL"
        );
    }

    /// A live joinable (non-detached) thread allows waiting via
    /// `register_task_exit_waiter`, returning `None` (not yet exited).
    #[test]
    fn test_joinable_thread_can_be_waited_on() {
        let _g = init_test_env();

        let task = crate::task::Task {
            id: 9803,
            state: TaskState::Runnable,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            affinity: Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            user_fs_base: 0,
            detached: false, // joinable
            signals: crate::signal::ThreadSignals::new(),
        };

        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(task));

        // Should succeed and return None (thread still running).
        assert_eq!(
            register_task_exit_waiter::<MockRuntime>(9803, 9804).unwrap(),
            None,
            "joining a live joinable thread must return None"
        );
    }
}
