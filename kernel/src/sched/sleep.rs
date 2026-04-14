//! Timing and yield functions.

use crate::BootRuntime;
use crate::BootTasking;

use super::SCHEDULER;

use super::types::{ScheduleReason, Scheduler};

/// Cooperative yield: attempt to switch to the next runnable task.
///
/// Returns `true` if a context switch occurred **or** there is runnable work
/// in the queues (i.e. the CPU should stay awake). Returns `false` when all
/// queues are empty and the caller may safely halt (HLT / WFI).
pub fn yield_now<R: BootRuntime>() -> bool {
    use core::sync::atomic::{AtomicU64, Ordering};

    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    let cpu_idx = super::current_cpu_index::<R>();

    let (switch_params, has_work) = {
        let lock_start = rt.mono_ticks();
        let lock = SCHEDULER.lock();
        let ptr = lock.expect("Scheduler not initialized");
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
        let sp = sched.schedule_point(ScheduleReason::CooperativeYield);
        let work = sched.has_runnable_work(cpu_idx);
        super::record_sched_lock_hold::<R>(
            &super::PROF_SCHED_LOCK_YIELD_NOW_CALLS,
            &super::PROF_SCHED_LOCK_YIELD_NOW_US_TOTAL,
            &super::PROF_SCHED_LOCK_YIELD_NOW_US_MAX,
            lock_start,
        );
        (sp, work)
    };

    if let Some(switch) = switch_params {
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

        rt.irq_restore(_irq);
        return true; // switched
    }

    rt.irq_restore(_irq);
    has_work
}

/// True blocking sleep - puts task in sleep queue and reschedules
pub fn sleep_ticks<R: BootRuntime>(ticks: u64) {
    if ticks == 0 {
        // Zero sleep = just yield once
        yield_now::<R>();
        return;
    }

    let rt = crate::runtime::<R>();
    let _irq = rt.irq_disable();

    let switch_params = {
        let lock_start = rt.mono_ticks();
        let lock = SCHEDULER.lock();
        let ptr = lock.expect("Scheduler not initialized");
        let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };

        // Get current task ID
        let current_id = {
            let cpu = super::current_cpu_index::<R>();
            match sched.state.per_cpu.get(cpu).and_then(|pc| pc.current) {
                Some(id) => {
                    // crate::ktrace!(
                    //     "SCHED: CPU {} task {} sleeping for {} ticks",
                    //     cpu, id, ticks
                    // );
                    id
                }
                None => {
                    // No current task (shouldn't happen)
                    // crate::kerror!("SCHED: CPU {} sleeping without current task!", cpu);
                    return;
                }
            }
        };

        // Calculate wake time and add to sleep queue
        let wake_tick = super::TICK_COUNT.load(core::sync::atomic::Ordering::Relaxed) + ticks;
        sched
            .state
            .sleep_queue
            .entry(wake_tick)
            .or_default()
            .push(current_id);

        if let Some(mut task) = crate::task::registry::get_task_mut::<R>(current_id) {
            task.state = crate::task::TaskState::Blocked;
        }

        // Do NOT push current task to runq - it's now sleeping
        // Just call prepare_schedule to pick next task
        let switch = sched.prepare_schedule();
        super::record_sched_lock_hold::<R>(
            &super::PROF_SCHED_LOCK_SLEEP_TICKS_CALLS,
            &super::PROF_SCHED_LOCK_SLEEP_TICKS_US_TOTAL,
            &super::PROF_SCHED_LOCK_SLEEP_TICKS_US_MAX,
            lock_start,
        );
        switch
    };

    if let Some(switch) = switch_params {
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
        // crate::ktrace!("SCHED: task woke up on CPU");
    }

    rt.irq_restore(_irq);
}

pub fn sleep_until<R: BootRuntime>(deadline_ticks: u64) {
    let rt = crate::runtime::<R>();
    loop {
        let now = rt.mono_ticks();
        if now >= deadline_ticks {
            break;
        }

        let _irq = rt.irq_disable();
        let switch_params = {
            let lock = SCHEDULER.lock();
            let ptr = lock.expect("Scheduler not initialized");
            let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
            sched.schedule_point(ScheduleReason::SleepWait)
        };
        rt.irq_restore(_irq);

        if let Some(switch) = switch_params {
            unsafe {
                let _irq = rt.irq_disable();
                rt.tasking().activate_address_space(switch.to_aspace);

                rt.tasking().switch_with_tls(
                    &mut *switch.from_ctx,
                    &*switch.to_ctx,
                    switch.to_tid,
                    switch.from_user_fs_base,
                    switch.to_user_fs_base,
                );
                rt.irq_restore(_irq);
            }
        } else {
            // No switch occurred, spin briefly
            core::hint::spin_loop();
        }
    }
}

pub fn sleep_ms<R: BootRuntime>(ms: u64) {
    // 1 tick is 10ms (100Hz timer)
    // Round up to avoid undersleeping.
    // ms=0 converts to ticks=0, which sleep_ticks handles as a yield.
    let ticks = (ms + 9) / 10;
    sleep_ticks::<R>(ticks);
}
