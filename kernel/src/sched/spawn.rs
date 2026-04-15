//! Task and thread spawning functions.

use crate::task::{Affinity, ProcessInfo, StartupArg, Task, TaskId, TaskState};
use crate::{BootRuntime, BootTasking, UserEntry};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::SCHEDULER;
use super::types::{DEFAULT_TIMESLICE, Scheduler};
use core::sync::atomic::{AtomicUsize, Ordering};

const KERNEL_STACK_SIZE: usize = 65536;

fn boot_module_matches(name: &str, module_name: &str) -> bool {
    if module_name == name {
        return true;
    }

    module_name.rsplit('/').next().unwrap_or(module_name) == name.rsplit('/').next().unwrap_or(name)
}

fn current_parent_pid<R: BootRuntime>(sched: &Scheduler<R>) -> u32 {
    let cpu_idx = super::current_cpu_index::<R>();
    sched
        .state
        .per_cpu
        .get(cpu_idx)
        .and_then(|pc| pc.current)
        .and_then(|ctid| crate::task::registry::get_task::<R>(ctid))
        .and_then(|t| t.process_info.clone())
        .map(|pi| pi.lock().pid)
        .unwrap_or(0)
}

fn default_process_info(
    pid: u32,
    ppid: u32,
    space: crate::task::ProcessAddressSpace,
) -> alloc::sync::Arc<spin::Mutex<ProcessInfo>> {
    let console_node: alloc::sync::Arc<dyn crate::vfs::VfsNode> =
        alloc::sync::Arc::new(crate::vfs::devfs::ConsoleNode);
    let mut fd_table = crate::vfs::fd_table::FdTable::new();
    let _ = fd_table.insert_at(
        0,
        console_node.clone(),
        crate::vfs::OpenFlags::read_only(),
        "/dev/console".into(),
    );
    let _ = fd_table.insert_at(
        1,
        console_node.clone(),
        crate::vfs::OpenFlags::write_only(),
        "/dev/console".into(),
    );
    let _ = fd_table.insert_at(
        2,
        console_node,
        crate::vfs::OpenFlags::write_only(),
        "/dev/console".into(),
    );
    let is_session_leader = ppid == 0;
    alloc::sync::Arc::new(spin::Mutex::new(ProcessInfo {
        pid,
        lifecycle: crate::task::ProcessLifecycle::new(ppid, pid as TaskId),
        unix_compat: crate::task::ProcessUnixCompat::isolated(pid, is_session_leader),
        fd_table,
        namespace: crate::vfs::NamespaceRef::global(),
        cwd: alloc::string::String::from("/"),
        exec_path: alloc::string::String::new(),
        space,
    }))
}

fn inherit_process_info<R: BootRuntime>(
    pid: u32,
    ppid: u32,
    space: crate::task::ProcessAddressSpace,
) -> alloc::sync::Arc<spin::Mutex<ProcessInfo>> {
    let tid = crate::runtime::<R>().current_tid();
    let current_pinfo =
        crate::task::registry::get_task::<R>(tid).and_then(|t| t.process_info.clone());

    if let Some(parent_pi) = current_pinfo {
        let parent = parent_pi.lock();
        // Create the first-class Space object from the ProcessAddressSpace fields.
        alloc::sync::Arc::new(spin::Mutex::new(ProcessInfo {
            pid,
            lifecycle: crate::task::ProcessLifecycle::new(ppid, pid as TaskId),
            unix_compat: crate::task::ProcessUnixCompat::inherit(&parent.unix_compat),
            fd_table: parent.fd_table.clone(),
            namespace: parent.namespace.clone(),
            cwd: parent.cwd.clone(),
            exec_path: alloc::string::String::new(),
            space,
        }))
    } else {
        default_process_info(pid, ppid, space)
    }
}

// Global round-robin index for CPU selection
pub(crate) static RR_IDX: AtomicUsize = AtomicUsize::new(0);

/// Register `tid` in the owning process's thread-group list.
///
/// This is the **single, mandatory** registration point for every thread that
/// belongs to a user process.  All spawn paths (`spawn_user_thread`,
/// `spawn_user_task`, `boot_spawn_process`, `boot_spawn_process_ex`,
/// `spawn_process_from_path`) call this function so that exec-collapse and
/// thread-group exit can reliably enumerate every live thread via
/// `ProcessInfo::thread_ids`.
///
/// Duplicate-safe: the TID is only appended if not already present, so calling
/// this more than once for the same TID is harmless.
///
/// # Why this matters
/// Previously the registration in `spawn_user_thread` was inadvertently gated
/// on `tls_base != 0`.  Threads created without a TLS base (the common case for
/// many POSIX-style threads) were invisible to exec-collapse and thread-group
/// teardown.  Extracting the logic into this function ensures:
///   1. The invariant is expressed once, not scattered across callers.
///   2. Future spawn paths cannot forget to register by accident — they just
///      call `register_thread_in_process` after building the task.
pub(crate) fn register_thread_in_process(
    pinfo: &Option<alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>>>,
    tid: crate::task::TaskId,
) {
    if let Some(pinfo) = pinfo {
        let mut pi = pinfo.lock();
        if !pi.lifecycle.thread_ids.contains(&tid) {
            pi.lifecycle.thread_ids.push(tid);
        }
    }
}

impl<R: BootRuntime> Scheduler<R> {
    fn pick_cpu_and_bringup(&mut self, affinity: Affinity, _trigger_smp: bool) -> usize {
        let rt = crate::runtime::<R>();
        let count = self.state.online_cpu_count;
        let idx = RR_IDX.fetch_add(1, Ordering::Relaxed);

        match affinity {
            Affinity::Pinned(cpu) => cpu,
            Affinity::Any => {
                // will be brought up manually when needed.
                //     if let Some(next_cpu_id) = rt.next_offline_cpu() {
                //         let target_cpu = next_cpu_id.0 as usize;
                //         unsafe {
                //             let _ = rt.start_cpu(next_cpu_id, crate::kernel_secondary_entry::<R>, target_cpu);
                //         }
                //         return target_cpu;
                //     }
                // }
                let _ = (rt, count); // Suppress unused variable warning
                self.state.pick_online_cpu_excluding_bsp(idx)
            }
        }
    }
    pub fn spawn(
        &mut self,
        entry: extern "C" fn(usize) -> !,
        arg: StartupArg,
        priority: crate::task::TaskPriority,
        affinity: Affinity,
    ) -> TaskId {
        let rt = crate::runtime::<R>();
        let id = self.next_id;
        self.next_id += 1;

        let layout = alloc::alloc::Layout::from_size_align(KERNEL_STACK_SIZE, 8).unwrap();
        let stack_base = unsafe { alloc::alloc::alloc(layout) };
        if stack_base.is_null() {
            panic!("Failed to allocate stack for task {}", id);
        }
        let stack_top = (stack_base as u64) + KERNEL_STACK_SIZE as u64;

        let ctx = rt
            .tasking()
            .init_kernel_context(entry, stack_top, arg.to_raw());

        // Determine target CPU: Balanced among online CPUs.
        let target_cpu = self.pick_cpu_and_bringup(affinity, false);
        crate::kdebug!("SCHED: Task {} assigned to CPU {}", id, target_cpu);

        // Push to target CPU's run queue
        let cpu_count = self.state.per_cpu.len(); // Should match rt.cpu_count()
        let safe_cpu = if target_cpu < cpu_count {
            target_cpu
        } else {
            0
        };

        let task: Task<R> = Task {
            id,
            state: TaskState::Runnable,
            priority,
            kstack_base: stack_base,
            kstack_size: KERNEL_STACK_SIZE,
            kstack_top: stack_top,
            ctx,
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
            timeslice_remaining: DEFAULT_TIMESLICE,
            affinity,
            last_cpu: Some(safe_cpu),
            name: [0; 32],
            name_len: 0,
            process_info: None,
            enqueued_at_tick: super::TICK_COUNT.load(Ordering::Relaxed),
            base_priority: priority,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        let sched_fields = crate::sched::state::TaskSchedFields {
            tid: task.id,
            runq_location: None,
        };
        self.state.insert_task(sched_fields);
        crate::task::registry::get_registry::<R>().insert(alloc::boxed::Box::new(task));
        self.state.enqueue_task(safe_cpu, priority as usize, id);

        // Under the scheduler lock we only mark the target CPU dirty.
        // The caller sends the IPI after unlocking so the target CPU's
        // resched handler can take the lock immediately.
        if safe_cpu == super::current_cpu_index::<R>() {
            self.state.per_cpu[safe_cpu].need_resched = true;
        } else {
            super::GLOBAL_NEED_RESCHED[safe_cpu].store(true, core::sync::atomic::Ordering::Release);
            super::DIAG_IPI_SENT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            crate::kdebug!(
                "SCHED: Deferred Resched IPI to CPU {} for task {}",
                safe_cpu,
                id
            );
        }

        let parent_tid = self.state.per_cpu[super::current_cpu_index::<R>()].current;
        // Link affinity and initial location
        if let Affinity::Pinned(cpu) = affinity {}
        // Initial location matches target runq

        id
    }

    pub fn spawn_user_thread(
        &mut self,
        entry: usize,
        stack: usize,
        arg: StartupArg,
        stack_info: abi::types::StackInfo,
        priority: crate::task::TaskPriority,
        affinity: Affinity,
        tls_base: u64,
        detached: bool,
    ) -> TaskId {
        let rt = crate::runtime::<R>();
        let id = self.next_id;
        self.next_id += 1;

        let layout = alloc::alloc::Layout::from_size_align(KERNEL_STACK_SIZE, 8).unwrap();
        let stack_base = unsafe { alloc::alloc::alloc(layout) };
        if stack_base.is_null() {
            panic!("Failed to allocate kernel stack for user thread {}", id);
        }
        let kstack_top = (stack_base as u64) + KERNEL_STACK_SIZE as u64;

        let aspace = rt.tasking().active_address_space();

        // Inherit mappings and process_info from the current process (not the
        // current task) so that the canonical VM state is always sourced from
        // Process rather than from an arbitrary thread's cached copy.
        let parent_pinfo =
            if let Some(current_id) = self.state.per_cpu[super::current_cpu_index::<R>()].current {
                crate::task::registry::get_task::<R>(current_id)
                    .and_then(|parent| parent.process_info.clone())
            } else {
                None
            };

        // Clone the mappings Arc from the parent process (same underlying
        // MappingList object).  Fall back to an empty list only when there is
        // no parent process (should not happen for user threads).
        let mappings = parent_pinfo
            .as_ref()
            .map(|pi| pi.lock().space.mappings.clone())
            .unwrap_or_else(|| {
                alloc::sync::Arc::new(spin::Mutex::new(crate::memory::mappings::MappingList::new()))
            });

        let spec = crate::UserTaskSpec {
            entry: entry as u64,
            stack_top: stack as u64,
            aspace,
            arg: arg.to_raw(),
        };

        let ctx = rt.tasking().init_user_context(spec, kstack_top);

        let target_cpu = match affinity {
            Affinity::Pinned(cpu) => cpu,
            Affinity::Any => super::current_cpu_index::<R>(),
        };
        crate::kdebug!(
            "SCHED: Task {} (user thread) assigned to CPU {}",
            id,
            target_cpu
        );

        // Push to target CPU's run queue
        let cpu_count = self.state.per_cpu.len();
        let safe_cpu = if target_cpu < cpu_count {
            target_cpu
        } else {
            0
        };

        let task: Task<R> = Task {
            id,
            state: TaskState::Runnable,
            priority,
            kstack_base: stack_base,
            kstack_size: KERNEL_STACK_SIZE,
            kstack_top,
            ctx,
            aspace,
            simd: crate::simd::SimdState::new(rt),
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            stack_info: Some(stack_info),
            mappings,
            timeslice_remaining: DEFAULT_TIMESLICE,
            affinity,
            last_cpu: Some(safe_cpu),
            name: [0; 32],
            name_len: 0,
            process_info: parent_pinfo,
            enqueued_at_tick: super::TICK_COUNT.load(Ordering::Relaxed),
            base_priority: priority,
            user_fs_base: tls_base,
            detached,
            signals: crate::signal::ThreadSignals::new(),
        };

        // Register this thread's TID in the owning process so exec and exit
        // can enumerate all threads.  Delegated to the central helper so that
        // every spawn path enforces the invariant identically and future callers
        // cannot accidentally skip registration.
        register_thread_in_process(&task.process_info, id);

        let sched_fields = crate::sched::state::TaskSchedFields {
            tid: task.id,
            runq_location: None,
        };
        self.state.insert_task(sched_fields);
        crate::task::registry::get_registry::<R>().insert(alloc::boxed::Box::new(task));
        self.state.enqueue_task(safe_cpu, priority as usize, id);

        // Under the scheduler lock we only mark the target CPU dirty.
        if safe_cpu == super::current_cpu_index::<R>() {
            self.state.per_cpu[safe_cpu].need_resched = true;
        } else {
            super::GLOBAL_NEED_RESCHED[safe_cpu].store(true, core::sync::atomic::Ordering::Release);
            crate::kdebug!(
                "SCHED: Deferred Resched IPI to CPU {} for task {}",
                safe_cpu,
                id
            );
        }

        let parent_tid = self.state.per_cpu[super::current_cpu_index::<R>()].current;
        // Link affinity and initial location
        if let Affinity::Pinned(cpu) = affinity {}
        // Initial location matches target runq

        id
    }

    pub fn spawn_user_task(
        &mut self,
        entry: UserEntry,
        aspace: <R::Tasking as BootTasking>::AddressSpace,
        stack_info: abi::types::StackInfo,
        regions: alloc::vec::Vec<abi::vm::VmRegionInfo>,
        priority: crate::task::TaskPriority,
        affinity: Affinity,
    ) -> Option<TaskId> {
        let rt = crate::runtime::<R>();
        let id = self.next_id;

        self.next_id += 1;
        let layout = alloc::alloc::Layout::from_size_align(KERNEL_STACK_SIZE, 8).unwrap();
        let stack_base = unsafe { alloc::alloc::alloc(layout) };
        if stack_base.is_null() {
            return None;
        }
        let stack_top = (stack_base as u64) + KERNEL_STACK_SIZE as u64;

        let user_entry = alloc::boxed::Box::new(entry);
        let entry_ptr = alloc::boxed::Box::into_raw(user_entry) as usize;

        let ctx =
            rt.tasking()
                .init_kernel_context(user_thread_trampoline::<R>, stack_top, entry_ptr);

        let mapping_list = crate::memory::mappings::MappingList { regions };
        let ppid = current_parent_pid::<R>(self);
        // Create the mappings Arc once — both the Process and the Thread hold a
        // clone of the same Arc so the scheduler's per-CPU CURRENT_MAPPINGS cache
        // works without locking the Process mutex on every context switch.
        let mappings_arc = alloc::sync::Arc::new(spin::Mutex::new(mapping_list));
        // Derive the process-owned address-space token from the typed aspace handle.
        let aspace_raw = rt.tasking().aspace_to_raw(aspace);
        let pinfo = default_process_info(id as u32, ppid, crate::task::ProcessAddressSpace::from_parts(mappings_arc.clone(), aspace_raw));

        let target_cpu = self.pick_cpu_and_bringup(affinity, true);
        // Push to target CPU's run queue
        let cpu_count = self.state.per_cpu.len();
        let safe_cpu = if target_cpu < cpu_count {
            target_cpu
        } else {
            0
        };

        let task: Task<R> = Task {
            id,
            state: TaskState::Runnable,
            priority,
            kstack_base: stack_base,
            kstack_size: KERNEL_STACK_SIZE,
            kstack_top: stack_top,
            ctx,
            aspace,
            simd: crate::simd::SimdState::new(rt),
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            stack_info: Some(stack_info),
            mappings: mappings_arc,
            timeslice_remaining: DEFAULT_TIMESLICE,
            affinity,
            last_cpu: Some(safe_cpu),
            name: [0; 32],
            name_len: 0,
            process_info: Some(pinfo),
            enqueued_at_tick: super::TICK_COUNT.load(Ordering::Relaxed),
            base_priority: priority,
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };

        let sched_fields = crate::sched::state::TaskSchedFields {
            tid: task.id,
            runq_location: None,
        };
        self.state.insert_task(sched_fields);
        crate::task::registry::get_registry::<R>().insert(alloc::boxed::Box::new(task));
        self.state.enqueue_task(safe_cpu, priority as usize, id);

        // Under the scheduler lock we only mark the target CPU dirty.
        if safe_cpu == super::current_cpu_index::<R>() {
            self.state.per_cpu[safe_cpu].need_resched = true;
        } else {
            super::GLOBAL_NEED_RESCHED[safe_cpu].store(true, core::sync::atomic::Ordering::Release);
            crate::kdebug!(
                "SCHED: Deferred Resched IPI to CPU {} for task {}",
                safe_cpu,
                id
            );
        }

        let parent_tid = self.state.per_cpu[super::current_cpu_index::<R>()].current;
        // Link affinity and initial location
        if let Affinity::Pinned(cpu) = affinity {}
        // Initial location matches target runq

        Some(id)
    }
}

pub fn spawn<R: BootRuntime>(
    entry: extern "C" fn(usize) -> !,
    arg: StartupArg,
    priority: crate::task::TaskPriority,
    affinity: crate::task::Affinity,
) -> TaskId {
    let rt = crate::runtime::<R>();
    let current_cpu = super::current_cpu_index::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
    let id = sched.spawn(entry, arg, priority, affinity);
    drop(lock);
    nudge_spawned_task::<R>(current_cpu, id);
    rt.irq_restore(_irq);
    id
}

pub fn spawn_with_priority<R: BootRuntime>(
    entry: extern "C" fn(usize) -> !,
    arg: StartupArg,
    priority: crate::task::TaskPriority,
) -> TaskId {
    spawn::<R>(entry, arg, priority, crate::task::Affinity::Any)
}

pub unsafe fn spawn_user_thread<R: BootRuntime>(
    entry: usize,
    stack: usize,
    arg: StartupArg,
    stack_info: abi::types::StackInfo,
    priority: crate::task::TaskPriority,
) -> TaskId {
    spawn_user_thread_ex::<R>(entry, stack, arg, stack_info, priority, 0, false)
}

/// Extended version of `spawn_user_thread` with explicit TLS base and detached flag.
pub unsafe fn spawn_user_thread_ex<R: BootRuntime>(
    entry: usize,
    stack: usize,
    arg: StartupArg,
    stack_info: abi::types::StackInfo,
    priority: crate::task::TaskPriority,
    tls_base: u64,
    detached: bool,
) -> TaskId {
    let rt = crate::runtime::<R>();
    let current_cpu = super::current_cpu_index::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
    let id = sched.spawn_user_thread(
        entry,
        stack,
        arg,
        stack_info,
        priority,
        crate::task::Affinity::Any,
        tls_base,
        detached,
    );
    drop(lock);
    nudge_spawned_task::<R>(current_cpu, id);
    rt.irq_restore(_irq);
    id
}

pub unsafe fn spawn_user_task_full<R: BootRuntime>(
    entry: UserEntry,
    aspace: <R::Tasking as BootTasking>::AddressSpace,
    stack_info: abi::types::StackInfo,
    regions: alloc::vec::Vec<abi::vm::VmRegionInfo>,
    priority: crate::task::TaskPriority,
) -> Option<TaskId> {
    let rt = crate::runtime::<R>();
    let current_cpu = super::current_cpu_index::<R>();
    let _irq = rt.irq_disable();
    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };
    let id = sched.spawn_user_task(
        entry,
        aspace,
        stack_info,
        regions,
        priority,
        crate::task::Affinity::Any,
    );
    drop(lock);
    if let Some(id) = id {
        nudge_spawned_task::<R>(current_cpu, id);
        rt.irq_restore(_irq);
        Some(id)
    } else {
        rt.irq_restore(_irq);
        None
    }
}

fn nudge_spawned_task<R: BootRuntime>(current_cpu: usize, id: TaskId) {
    let Some(target_cpu) = crate::task::registry::get_task::<R>(id).and_then(|task| task.last_cpu)
    else {
        return;
    };

    if target_cpu != current_cpu {
        crate::kdebug!(
            "SCHED: Sending post-unlock Resched IPI to CPU {} for task {}",
            target_cpu,
            id
        );
        crate::runtime::<R>().send_ipi(target_cpu, 0x30);
    }
}

/// Boot-only helper: spawn a process from a boot module by name at normal priority.
///
/// This is a **boot-time convenience**.  It locates the named module in the
/// static boot module table (`BootRuntime::modules`) and launches it.  It must
/// **not** be used for runtime process creation; call
/// [`spawn_process_from_path`] instead.
pub unsafe fn boot_spawn_process<R: BootRuntime>(name: &str, arg: StartupArg) -> Option<TaskId> {
    unsafe { boot_spawn_process_with_priority::<R>(name, arg, crate::task::TaskPriority::Normal) }
}

/// Boot-only helper: spawn a process from a boot module at a given priority.
///
/// Same as [`boot_spawn_process`] but with an explicit priority.  Scoped to
/// boot use; runtime callers should use [`spawn_process_from_path`].
pub unsafe fn boot_spawn_process_with_priority<R: BootRuntime>(
    name: &str,
    arg: StartupArg,
    priority: crate::task::TaskPriority,
) -> Option<TaskId> {
    let rt = crate::runtime::<R>();
    let current_cpu = super::current_cpu_index::<R>();
    let modules = rt.modules();
    let basename = name.rsplit('/').next().unwrap_or(name);
    let module = modules.iter().find(|m| m.name.contains(basename))?;

    let aspace = rt.tasking().make_user_address_space();

    let (mut entry, stack_info, regions, aux_info) =
        crate::task::loader::load_module(rt, aspace, module)?;
    entry.arg0 = arg.to_raw();

    let _irq = rt.irq_disable();

    let affinity = if name.contains("virtio_sound") || name.contains("beeper") {
        crate::task::Affinity::Pinned(0)
    } else if name == "bloom" {
        if rt.cpu_total_count() > 1 {
            crate::task::Affinity::Pinned(1)
        } else {
            crate::task::Affinity::Any
        }
    } else if name.contains("/sh") {
        if rt.cpu_total_count() > 1 {
            crate::task::Affinity::Pinned(rt.cpu_total_count() - 1)
        } else {
            crate::task::Affinity::Any
        }
    } else {
        crate::task::Affinity::Any
    };

    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &mut *(ptr as *mut Scheduler<R>) };

    let id = sched.spawn_user_task(entry, aspace, stack_info, regions, priority, affinity)?;

    // Determine parent PID from the current task's ProcessInfo
    let ppid = current_parent_pid::<R>(sched);

    // Retrieve the mappings Arc from the task that was just created so the
    // Process and Thread share the same underlying MappingList.
    let task_mappings = crate::task::registry::get_task::<R>(id)
        .map(|t| t.mappings.clone())
        .unwrap_or_else(|| {
            alloc::sync::Arc::new(spin::Mutex::new(crate::memory::mappings::MappingList::new()))
        });

    // Derive the process-owned address-space token (raw u64) from the handle.
    let aspace_raw = rt.tasking().aspace_to_raw(aspace);

    // Create per-process identity
    let pinfo = inherit_process_info::<R>(id as u32, ppid, crate::task::ProcessAddressSpace::from_parts(task_mappings, aspace_raw));
    {
        let page_size = rt.page_size() as u64;
        let mut lock = pinfo.lock();
        lock.unix_compat.argv = alloc::vec![module.name.as_bytes().to_vec()];
        lock.unix_compat.auxv = crate::task::exec::build_auxv(&aux_info, page_size);
        lock.exec_path = alloc::format!("/boot/{}", module.name);
    }

    // Store name, process_info, and initial TLS thread pointer on the task struct.
    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(id) {
        let bytes = module.name.as_bytes();
        let len = bytes.len().min(32);
        task.name[..len].copy_from_slice(&bytes[..len]);
        task.name_len = len as u8;
        task.process_info = Some(pinfo);
        // Apply initial TLS base (FS_BASE on x86_64) for the new process's main thread.
        // Zero means no PT_TLS segment was found; FS_BASE starts at its default state.
        task.user_fs_base = aux_info.tls_tp;
    }

    drop(lock);
    nudge_spawned_task::<R>(current_cpu, id);
    rt.irq_restore(_irq);
    Some(id)
}

/// Stdio specification for a single stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdioSpec {
    /// Inherit parent's handle (child gets a dup of the parent's fd).
    Inherit,
    /// Attach to null sink/source.
    Null,
    /// Create a pipe; returns the pipe_id for the parent's end.
    Pipe,
    /// Clone the specified parent fd into this stdio slot.
    Fd(u32),
}

/// Populate the fd_table slots 0, 1, 2 in `fd_table` based on the given specs.
///
/// Returns the pipe IDs allocated for piped stdin/stdout/stderr (0 when not piped).
/// The parent uses these pipe IDs with the legacy `SYS_PIPE_*` syscalls; the child
/// reads/writes through the fd_table nodes which share the same underlying pipe.
fn setup_stdio_fds<R: BootRuntime>(
    fd_table: &mut crate::vfs::fd_table::FdTable,
    stdin_spec: StdioSpec,
    stdout_spec: StdioSpec,
    stderr_spec: StdioSpec,
) -> (u64, u64, u64) {
    use crate::vfs::{OpenFlags, VfsNode};
    use alloc::sync::Arc;

    let console: Arc<dyn VfsNode> = Arc::new(crate::vfs::devfs::ConsoleNode);
    let null: Arc<dyn VfsNode> = Arc::new(crate::vfs::devfs::NullNode);

    // Helper: inherit parent's fd by cloning the node Arc.
    let inherited_node = |fd: u32| -> Option<(Arc<dyn VfsNode>, OpenFlags)> {
        let tid = crate::runtime::<R>().current_tid();
        crate::task::registry::get_task::<R>(tid)
            .and_then(|task| task.process_info.clone())
            .and_then(|pi| {
                let lock = pi.lock();
                lock.fd_table
                    .get(fd)
                    .ok()
                    .map(|f| (f.node.clone(), *f.status_flags.lock()))
            })
    };

    let mut stdin_pipe: u64 = 0;
    let mut stdout_pipe: u64 = 0;
    let mut stderr_pipe: u64 = 0;

    // fd 0 — stdin
    match stdin_spec {
        StdioSpec::Inherit => {
            if let Some((node, flags)) = inherited_node(0) {
                let _ = fd_table.insert_at(0, node, flags, "/dev/console".into());
            } else {
                let _ = fd_table.insert_at(
                    0,
                    console.clone(),
                    OpenFlags::read_only(),
                    "/dev/console".into(),
                );
            }
        }
        StdioSpec::Null => {
            let _ = fd_table.insert_at(0, null.clone(), OpenFlags::read_only(), "/dev/null".into());
        }
        StdioSpec::Pipe => {
            // Create the raw pipe (readers=1, writers=1).  The child's fd 0 is
            // the read end; the parent retains the write end via stdin_pipe.
            let id = crate::ipc::pipe::create(4096, 0);
            if let Some(read_node) = crate::ipc::pipe::read_node_for_id(id) {
                let _ = fd_table.insert_at(
                    0,
                    read_node,
                    OpenFlags::read_only(),
                    alloc::format!("pipe:{}", id),
                );
            }
            stdin_pipe = id;
        }
        StdioSpec::Fd(fd) => {
            if let Some((node, flags)) = inherited_node(fd) {
                let path = alloc::format!("fd:{}", fd);
                let _ = fd_table.insert_at(0, node, flags, path);
            } else {
                let _ =
                    fd_table.insert_at(0, null.clone(), OpenFlags::read_only(), "/dev/null".into());
            }
        }
    }

    // fd 1 — stdout
    match stdout_spec {
        StdioSpec::Inherit => {
            if let Some((node, flags)) = inherited_node(1) {
                let _ = fd_table.insert_at(1, node, flags, "/dev/console".into());
            } else {
                let _ = fd_table.insert_at(
                    1,
                    console.clone(),
                    OpenFlags::write_only(),
                    "/dev/console".into(),
                );
            }
        }
        StdioSpec::Null => {
            let _ =
                fd_table.insert_at(1, null.clone(), OpenFlags::write_only(), "/dev/null".into());
        }
        StdioSpec::Pipe => {
            let id = crate::ipc::pipe::create(4096, 0);
            if let Some(write_node) = crate::ipc::pipe::write_node_for_id(id) {
                let _ = fd_table.insert_at(
                    1,
                    write_node,
                    OpenFlags::write_only(),
                    alloc::format!("pipe:{}", id),
                );
            }
            stdout_pipe = id;
        }
        StdioSpec::Fd(fd) => {
            if let Some((node, flags)) = inherited_node(fd) {
                let path = alloc::format!("fd:{}", fd);
                let _ = fd_table.insert_at(1, node, flags, path);
            } else {
                let _ = fd_table.insert_at(
                    1,
                    null.clone(),
                    OpenFlags::write_only(),
                    "/dev/null".into(),
                );
            }
        }
    }

    // fd 2 — stderr
    match stderr_spec {
        StdioSpec::Inherit => {
            if let Some((node, flags)) = inherited_node(2) {
                let _ = fd_table.insert_at(2, node, flags, "/dev/console".into());
            } else {
                let _ =
                    fd_table.insert_at(2, console, OpenFlags::write_only(), "/dev/console".into());
            }
        }
        StdioSpec::Null => {
            let _ = fd_table.insert_at(2, null, OpenFlags::write_only(), "/dev/null".into());
        }
        StdioSpec::Pipe => {
            let id = crate::ipc::pipe::create(4096, 0);
            if let Some(write_node) = crate::ipc::pipe::write_node_for_id(id) {
                let _ = fd_table.insert_at(
                    2,
                    write_node,
                    OpenFlags::write_only(),
                    alloc::format!("pipe:{}", id),
                );
            }
            stderr_pipe = id;
        }
        StdioSpec::Fd(fd) => {
            if let Some((node, flags)) = inherited_node(fd) {
                let path = alloc::format!("fd:{}", fd);
                let _ = fd_table.insert_at(2, node, flags, path);
            } else {
                let _ = fd_table.insert_at(2, null, OpenFlags::write_only(), "/dev/null".into());
            }
        }
    }

    (stdin_pipe, stdout_pipe, stderr_pipe)
}
/// Result of an enhanced spawn: child tid + pipe IDs for piped stdio.
#[derive(Debug, Clone)]
pub struct SpawnExResult {
    pub child_tid: TaskId,
    pub child_pid: u32,
    /// Parent's fd for stdin (parent writes to this fd). 0 if not piped.
    pub stdin_pipe: u64,
    /// Parent's fd for stdout (parent reads from this fd). 0 if not piped.
    pub stdout_pipe: u64,
    /// Parent's fd for stderr (parent reads from this fd). 0 if not piped.
    pub stderr_pipe: u64,
}

/// Boot-only helper: enhanced process spawn from a boot module with explicit
/// argv, env, and stdio piping.
///
/// This function looks up the executable in the boot module table.  It is
/// scoped to boot/module-launch use only.  For runtime process creation use
/// [`spawn_process_from_path`] instead.
///
/// # Safety
/// Must be called with scheduler lock expectations satisfied.
pub unsafe fn boot_spawn_process_ex<R: BootRuntime>(
    name: &str,
    argv: Vec<Vec<u8>>,
    env: BTreeMap<Vec<u8>, Vec<u8>>,
    stdin_spec: StdioSpec,
    stdout_spec: StdioSpec,
    stderr_spec: StdioSpec,
    boot_arg: u64,
    inherited_handles: Vec<u64>,
    cwd: Option<alloc::string::String>,
    fd_remap: Vec<abi::types::FdRemap>,
) -> Result<SpawnExResult, abi::errors::Errno> {
    let rt = crate::runtime::<R>();
    let current_cpu = super::current_cpu_index::<R>();
    let modules = rt.modules();
    let module = modules
        .iter()
        .find(|m| boot_module_matches(name, m.name))
        .ok_or(abi::errors::Errno::ENOENT)?;

    let aspace = rt.tasking().make_user_address_space();

    let (mut entry, stack_info, regions, aux_info) =
        crate::task::loader::load_module(rt, aspace, module).ok_or(abi::errors::Errno::ENOEXEC)?;
    entry.arg0 = boot_arg as usize;

    let _irq = rt.irq_disable();

    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &mut *(ptr as *mut super::types::Scheduler<R>) };

    let id = sched
        .spawn_user_task(
            entry,
            aspace,
            stack_info,
            regions,
            crate::task::TaskPriority::Normal,
            crate::task::Affinity::Any,
        )
        .ok_or(abi::errors::Errno::EAGAIN)?;

    // Determine parent PID
    let cpu_idx = super::current_cpu_index::<R>();
    let ppid = sched
        .state
        .per_cpu
        .get(cpu_idx)
        .and_then(|pc| pc.current)
        .and_then(|ctid| crate::task::registry::get_task::<R>(ctid))
        .and_then(|t| t.process_info.clone())
        .map(|pi| pi.lock().pid)
        .unwrap_or(0);

    // Use provided argv, or fall back to module name
    let final_argv = if argv.is_empty() {
        alloc::vec![module.name.as_bytes().to_vec()]
    } else {
        argv
    };

    // Populate stdio fds in the child's fd_table.
    let tid = crate::runtime::<R>().current_tid();
    let parent_pinfo =
        crate::task::registry::get_task::<R>(tid).and_then(|t| t.process_info.clone());

    let mut fd_table = if let Some(parent_pi) = &parent_pinfo {
        parent_pi.lock().fd_table.clone()
    } else {
        crate::vfs::fd_table::FdTable::new()
    };

    let (stdin_pipe_id, stdout_pipe_id, stderr_pipe_id) =
        setup_stdio_fds::<R>(&mut fd_table, stdin_spec, stdout_spec, stderr_spec);

    // Step 6b: Apply explicit FD remappings.
    // These take precedence over stdio/inherited defaults for the same slots.
    for remap in fd_remap {
        if let Err(e) = fd_table.dup2(remap.src_fd, remap.dst_fd) {
            crate::kprintln!(
                "SPAWN: FD remap failed: {} -> {} (errno {:?})",
                remap.src_fd,
                remap.dst_fd,
                e
            );
        }
    }

    // Open the parent-side pipe ends in the parent's fd table so the parent
    // can communicate with the child via normal file descriptors.
    //
    // For stdin PIPE:  parent holds the WRITE end (fd is returned as stdin_pipe).
    // For stdout PIPE: parent holds the READ end (fd is returned as stdout_pipe).
    // For stderr PIPE: parent holds the READ end (fd is returned as stderr_pipe).
    //
    // If there is no parent process, or an end cannot be opened, the pipe will
    // still work from the child's side (it will see EOF when the write end is
    // never written to / the read end is never read from).
    let mut parent_stdin_fd: u64 = 0;
    let mut parent_stdout_fd: u64 = 0;
    let mut parent_stderr_fd: u64 = 0;
    if let Some(parent_pi) = &parent_pinfo {
        let mut plk = parent_pi.lock();
        if stdin_pipe_id != 0 {
            if let Some(write_node) = crate::ipc::pipe::write_node_for_id(stdin_pipe_id) {
                if let Ok(fd) = plk.fd_table.open(
                    write_node,
                    crate::vfs::OpenFlags::write_only(),
                    alloc::format!("pipe:{}", stdin_pipe_id),
                ) {
                    parent_stdin_fd = fd as u64;
                }
            }
        }
        if stdout_pipe_id != 0 {
            if let Some(read_node) = crate::ipc::pipe::read_node_for_id(stdout_pipe_id) {
                if let Ok(fd) = plk.fd_table.open(
                    read_node,
                    crate::vfs::OpenFlags::read_only(),
                    alloc::format!("pipe:{}", stdout_pipe_id),
                ) {
                    parent_stdout_fd = fd as u64;
                }
            }
        }
        if stderr_pipe_id != 0 {
            if let Some(read_node) = crate::ipc::pipe::read_node_for_id(stderr_pipe_id) {
                if let Ok(fd) = plk.fd_table.open(
                    read_node,
                    crate::vfs::OpenFlags::read_only(),
                    alloc::format!("pipe:{}", stderr_pipe_id),
                ) {
                    parent_stderr_fd = fd as u64;
                }
            }
        }
    }

    // Retrieve the mappings Arc from the task so Process and Thread share the
    // same underlying MappingList.
    let task_mappings = crate::task::registry::get_task::<R>(id)
        .map(|t| t.mappings.clone())
        .unwrap_or_else(|| {
            alloc::sync::Arc::new(spin::Mutex::new(crate::memory::mappings::MappingList::new()))
        });

    // Derive the process-owned address-space token (raw u64) from the handle.
    let aspace_raw = rt.tasking().aspace_to_raw(aspace);

    let unix_compat = if let Some(parent_pi) = &parent_pinfo {
        let parent = parent_pi.lock();
        let mut uc = crate::task::ProcessUnixCompat::inherit(&parent.unix_compat);
        uc.argv = final_argv;
        uc.env = env;
        uc.auxv = crate::task::exec::build_auxv(&aux_info, rt.page_size() as u64);
        uc
    } else {
        let mut uc = crate::task::ProcessUnixCompat::isolated(id as u32, true);
        uc.argv = final_argv;
        uc.env = env;
        uc.auxv = crate::task::exec::build_auxv(&aux_info, rt.page_size() as u64);
        uc
    };

    // Create per-process identity with provided argv & env
    let pinfo = alloc::sync::Arc::new(spin::Mutex::new(ProcessInfo {
        pid: id as u32,
        lifecycle: crate::task::ProcessLifecycle::new(ppid, id),
        unix_compat,
        fd_table,
        namespace: crate::vfs::NamespaceRef::global(),
        cwd: if let Some(explicit_cwd) = cwd {
            explicit_cwd
        } else if let Some(parent_pi) = &parent_pinfo {
            parent_pi.lock().cwd.clone()
        } else {
            alloc::string::String::from("/")
        },
        exec_path: alloc::format!("/boot/{}", module.name),
        space: crate::task::ProcessAddressSpace::from_parts(task_mappings, aspace_raw),
    }));

    // Store name, process_info, and initial TLS thread pointer on the task struct.
    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(id) {
        let bytes = module.name.as_bytes();
        let len = bytes.len().min(32);
        task.name[..len].copy_from_slice(&bytes[..len]);
        task.name_len = len as u8;
        task.process_info = Some(pinfo);
        // Apply initial TLS base (FS_BASE on x86_64) for the new process's main thread.
        task.user_fs_base = aux_info.tls_tp;
    }

    drop(lock);
    nudge_spawned_task::<R>(current_cpu, id);
    rt.irq_restore(_irq);

    Ok(SpawnExResult {
        child_tid: id,
        child_pid: id as u32,
        stdin_pipe: parent_stdin_fd,
        stdout_pipe: parent_stdout_fd,
        stderr_pipe: parent_stderr_fd,
    })
}

/// General-purpose runtime process creation from a VFS path.
///
/// This is the **standard runtime process creation path** that follows the
/// ThingOS process model:
///
/// 1. Open the executable from the VFS (e.g. `/usr/bin/ls`).
/// 2. Build a new process object and initial thread.
/// 3. Apply inheritance/replacement for stdio, fds, cwd, and env.
/// 4. Schedule the new thread for execution.
///
/// Unlike the boot helpers ([`boot_spawn_process`], [`boot_spawn_process_ex`]),
/// this function does **not** consult the boot module table and has no
/// boot-specific assumptions.  It is the correct function to call for any
/// runtime `SYS_SPAWN_PROCESS_EX` invocation.
///
/// # Safety
/// Must be called with scheduler lock expectations satisfied.
pub unsafe fn spawn_process_from_path<R: BootRuntime>(
    path: &str,
    argv: Vec<Vec<u8>>,
    env: BTreeMap<Vec<u8>, Vec<u8>>,
    stdin_spec: StdioSpec,
    stdout_spec: StdioSpec,
    stderr_spec: StdioSpec,
    _boot_arg: u64,
    inherited_handles: Vec<u64>,
    cwd: Option<alloc::string::String>,
    fd_remap: Vec<abi::types::FdRemap>,
) -> Result<SpawnExResult, abi::errors::Errno> {
    // Step 1: Open the executable from the VFS.
    let node = crate::vfs::mount::lookup(path).map_err(|_| abi::errors::Errno::ENOENT)?;

    let stat = node.stat().map_err(|e| e)?;
    if !stat.is_reg() {
        return Err(abi::errors::Errno::EACCES);
    }

    let size = stat.size as usize;
    if size > 64 * 1024 * 1024 {
        return Err(abi::errors::Errno::EFBIG);
    }

    // Step 2: Read the ELF bytes into kernel memory.
    let mut buffer = alloc::vec![0u8; size];
    let mut read_pos = 0;
    while read_pos < size {
        let n = node
            .read(read_pos as u64, &mut buffer[read_pos..])
            .map_err(|e| e)?;
        if n == 0 {
            break;
        }
        read_pos += n;
    }
    if read_pos < size {
        return Err(abi::errors::Errno::EIO);
    }

    // Step 3: Load the ELF into a fresh address space.
    let rt = crate::runtime::<R>();
    let current_cpu = super::current_cpu_index::<R>();
    let aspace = rt.tasking().make_user_address_space();

    // SAFETY: `load_module` is synchronous and does not retain the reference.
    let static_bytes: &'static [u8] = unsafe { core::mem::transmute(buffer.as_slice()) };
    let basename = path.rsplit('/').next().unwrap_or(path);
    // SAFETY: `load_module` is synchronous; `basename` outlives the call.
    let static_name: &'static str = unsafe { core::mem::transmute(basename) };
    let module_desc = crate::BootModuleDesc {
        name: static_name,
        cmdline: "",
        bytes: static_bytes,
        phys_start: 0,
        phys_end: 0,
        kind: crate::BootModuleKind::Elf,
    };

    let (entry, stack_info, regions, aux_info) =
        crate::task::loader::load_module(rt, aspace, &module_desc)
            .ok_or(abi::errors::Errno::ENOEXEC)?;

    // Step 4: Create the scheduler task for the new process's initial thread.
    let _irq = rt.irq_disable();

    let lock = SCHEDULER.lock();
    let ptr = lock.expect("Scheduler not initialized");
    let sched = unsafe { &mut *(ptr as *mut super::types::Scheduler<R>) };

    let id = sched
        .spawn_user_task(
            entry,
            aspace,
            stack_info,
            regions,
            crate::task::TaskPriority::Normal,
            crate::task::Affinity::Any,
        )
        .ok_or(abi::errors::Errno::EAGAIN)?;

    // Determine parent PID from the running task.
    let ppid = current_parent_pid::<R>(sched);

    // Step 5: Resolve argv — fall back to the executable basename.
    let final_argv = if argv.is_empty() {
        alloc::vec![basename.as_bytes().to_vec()]
    } else {
        argv
    };

    // Step 6: Inherit and set up stdio fds in the child's fd_table.
    let parent_tid = rt.current_tid();
    let parent_pinfo =
        crate::task::registry::get_task::<R>(parent_tid).and_then(|t| t.process_info.clone());

    let mut fd_table = if let Some(parent_pi) = &parent_pinfo {
        parent_pi.lock().fd_table.clone()
    } else {
        crate::vfs::fd_table::FdTable::new()
    };

    let (stdin_pipe_id, stdout_pipe_id, stderr_pipe_id) =
        setup_stdio_fds::<R>(&mut fd_table, stdin_spec, stdout_spec, stderr_spec);

    // Step 6b: Apply explicit FD remappings.
    for remap in fd_remap {
        if let Err(e) = fd_table.dup2(remap.src_fd, remap.dst_fd) {
            crate::kprintln!(
                "SPAWN: FD remap failed: {} -> {} (errno {:?})",
                remap.src_fd,
                remap.dst_fd,
                e
            );
        }
    }

    // Open the parent-side pipe ends in the parent's fd_table.
    let mut parent_stdin_fd: u64 = 0;
    let mut parent_stdout_fd: u64 = 0;
    let mut parent_stderr_fd: u64 = 0;
    if let Some(parent_pi) = &parent_pinfo {
        let mut plk = parent_pi.lock();
        if stdin_pipe_id != 0 {
            if let Some(write_node) = crate::ipc::pipe::write_node_for_id(stdin_pipe_id) {
                if let Ok(fd) = plk.fd_table.open(
                    write_node,
                    crate::vfs::OpenFlags::write_only(),
                    alloc::format!("pipe:{}", stdin_pipe_id),
                ) {
                    parent_stdin_fd = fd as u64;
                }
            }
        }
        if stdout_pipe_id != 0 {
            if let Some(read_node) = crate::ipc::pipe::read_node_for_id(stdout_pipe_id) {
                if let Ok(fd) = plk.fd_table.open(
                    read_node,
                    crate::vfs::OpenFlags::read_only(),
                    alloc::format!("pipe:{}", stdout_pipe_id),
                ) {
                    parent_stdout_fd = fd as u64;
                }
            }
        }
        if stderr_pipe_id != 0 {
            if let Some(read_node) = crate::ipc::pipe::read_node_for_id(stderr_pipe_id) {
                if let Ok(fd) = plk.fd_table.open(
                    read_node,
                    crate::vfs::OpenFlags::read_only(),
                    alloc::format!("pipe:{}", stderr_pipe_id),
                ) {
                    parent_stderr_fd = fd as u64;
                }
            }
        }
    }

    // Share the mapping list between the Thread and the Process.
    let task_mappings = crate::task::registry::get_task::<R>(id)
        .map(|t| t.mappings.clone())
        .unwrap_or_else(|| {
            alloc::sync::Arc::new(spin::Mutex::new(crate::memory::mappings::MappingList::new()))
        });

    let aspace_raw = rt.tasking().aspace_to_raw(aspace);

    let unix_compat = if let Some(parent_pi) = &parent_pinfo {
        let parent = parent_pi.lock();
        let mut uc = crate::task::ProcessUnixCompat::inherit(&parent.unix_compat);
        uc.argv = final_argv;
        uc.env = env;
        uc.auxv = crate::task::exec::build_auxv(&aux_info, rt.page_size() as u64);
        uc
    } else {
        let mut uc = crate::task::ProcessUnixCompat::isolated(id as u32, true);
        uc.argv = final_argv;
        uc.env = env;
        uc.auxv = crate::task::exec::build_auxv(&aux_info, rt.page_size() as u64);
        uc
    };

    // Step 7: Build the ProcessInfo for the new process.
    let pinfo = alloc::sync::Arc::new(spin::Mutex::new(ProcessInfo {
        pid: id as u32,
        lifecycle: crate::task::ProcessLifecycle::new(ppid, id),
        unix_compat,
        fd_table,
        namespace: crate::vfs::NamespaceRef::global(),
        cwd: if let Some(explicit_cwd) = cwd {
            explicit_cwd
        } else if let Some(parent_pi) = &parent_pinfo {
            parent_pi.lock().cwd.clone()
        } else {
            alloc::string::String::from("/")
        },
        exec_path: alloc::string::String::from(path),
        space: crate::task::ProcessAddressSpace::from_parts(task_mappings, aspace_raw),
    }));

    // Step 8: Attach the ProcessInfo to the new task and record its TLS base.
    if let Some(mut task) = crate::task::registry::get_task_mut::<R>(id) {
        let len = basename.len().min(32);
        task.name[..len].copy_from_slice(&basename.as_bytes()[..len]);
        task.name_len = len as u8;
        task.process_info = Some(pinfo);
        task.user_fs_base = aux_info.tls_tp;
    }

    // `inherited_handles` is reserved for future fd-inheritance; not yet wired.
    let _ = inherited_handles;

    drop(lock);
    nudge_spawned_task::<R>(current_cpu, id);
    rt.irq_restore(_irq);

    Ok(SpawnExResult {
        child_tid: id,
        child_pid: id as u32,
        stdin_pipe: parent_stdin_fd,
        stdout_pipe: parent_stdout_fd,
        stderr_pipe: parent_stderr_fd,
    })
}

pub extern "C" fn user_thread_trampoline<R: BootRuntime>(arg: usize) -> ! {
    let rt = crate::runtime::<R>();
    let entry_ptr = arg as *mut UserEntry;
    let entry = unsafe { *alloc::boxed::Box::from_raw(entry_ptr) };

    crate::kdebug!(
        "USER_TRAMPOLINE: PC=0x{:x} SP=0x{:x} ARG0=0x{:x}",
        entry.entry_pc,
        entry.user_sp,
        entry.arg0
    );

    unsafe { rt.tasking().enter_user(entry) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskPriority;

    // Re-use the shared mock runtime defined in `sched::tests` so that both
    // this module and `sched::mod` share a single `init_runtime` call and a
    // single `MockRuntime` type.  This prevents the "Runtime type mismatch" /
    // double-init panics that occur when each module defines its own mock.
    use crate::sched::tests::{MockRuntime, init_test_env};

    #[test]
    fn boot_module_match_requires_exact_basename() {
        assert!(boot_module_matches("ls", "/bin/ls"));
        assert!(boot_module_matches("/bin/ls", "/bin/ls"));
        assert!(!boot_module_matches("ls", "/bin/smallsh"));
        assert!(!boot_module_matches("/bin/ls", "/bin/smallsh"));
    }

    #[test]
    fn test_spawn_arg_semantics() {
        let _g = init_test_env();

        let mut sched = Scheduler::<MockRuntime>::new();
        sched.next_id = 5000;
        // Manually initialize PerCpu state for the mock
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(0); // Set a dummy current task ID for parent linking

        let cases = [
            (StartupArg::None, 0),
            (StartupArg::BootRegistry, 0x600000),
            (StartupArg::DeviceId(0x123), 0x123),
            (StartupArg::Raw(0x42), 0x42),
        ];

        for (arg, expected) in cases {
            let id = sched.spawn(mock_entry, arg, TaskPriority::Normal, Affinity::Any);
            let task = crate::task::registry::get_task::<MockRuntime>(id).unwrap();

            // In our MockTasking.init_kernel_context, we store arg in MockContext.0
            assert_eq!(task.ctx.0, expected);
            assert_eq!(arg.to_raw(), expected);
        }
    }

    /// Verify that `spawn_user_thread` correctly routes the startup argument
    /// into the task context so the entry function receives it in the first
    /// argument register (e.g. `rdi` on x86_64).
    ///
    /// The `MockRuntime::init_user_context` stores `spec.arg` directly in the
    /// mock context, so asserting `task.ctx.0 == expected` confirms the full
    /// pipeline:
    ///   `spawn_with_arg(entry, arg)`
    ///   → `SYS_SPAWN_THREAD` with `SpawnThreadReq { arg }`
    ///   → `StartupArg::Raw(arg)` passed to `spawn_user_thread`
    ///   → `UserTaskSpec { arg }` passed to `init_user_context`
    ///   → arg placed in first argument register on the target arch
    #[test]
    fn test_spawn_user_thread_arg() {
        let _g = init_test_env();

        let mut sched = Scheduler::<MockRuntime>::new();
        sched.next_id = 9000;
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        sched.state.per_cpu[0].current = Some(0);

        let stack_info = abi::types::StackInfo::default();

        let cases: &[(usize, usize)] = &[
            (0, 0),
            (0x1234, 0x1234),
            (0xDEAD_BEEF, 0xDEAD_BEEF),
            (usize::MAX, usize::MAX),
        ];

        for &(raw_arg, expected) in cases {
            let id = sched.spawn_user_thread(
                0x4000, // mock entry address
                0x8000, // mock user stack pointer
                StartupArg::Raw(raw_arg),
                stack_info,
                TaskPriority::Normal,
                Affinity::Any,
                0,     // tls_base
                false, // detached
            );
            let task = crate::task::registry::get_task::<MockRuntime>(id).unwrap();

            // MockRuntime::init_user_context stores spec.arg in MockContext.0
            assert_eq!(
                task.ctx.0, expected,
                "user thread arg mismatch for raw_arg={:#x}",
                raw_arg
            );
        }
    }

    extern "C" fn mock_entry(_arg: usize) -> ! {
        loop {}
    }

    // ── Thread-group membership invariant tests ───────────────────────────

    /// Helper: build a minimal `ProcessInfo` Arc with the given leader TID.
    fn make_process_info(
        leader: crate::task::TaskId,
    ) -> alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>> {
        alloc::sync::Arc::new(spin::Mutex::new(crate::task::ProcessInfo {
            pid: leader as u32,
            lifecycle: crate::task::ProcessLifecycle::new(1, leader),
            unix_compat: crate::task::ProcessUnixCompat::isolated(leader as u32, false),
            fd_table: crate::vfs::fd_table::FdTable::new(),
            namespace: crate::vfs::NamespaceRef::global(),
            cwd: alloc::string::String::from("/"),
            exec_path: alloc::string::String::new(),
            space: crate::task::ProcessAddressSpace::empty(),
        }))
    }

    /// Helper: build a minimal leader task backed by `pinfo` and insert it into
    /// the registry, then set it as the current task on CPU 0.
    fn setup_leader_task(
        sched: &mut Scheduler<MockRuntime>,
        leader_id: crate::task::TaskId,
        pinfo: alloc::sync::Arc<spin::Mutex<crate::task::ProcessInfo>>,
    ) {
        use crate::task::{TaskPriority, TaskState};
        let leader = crate::task::Task {
            id: leader_id,
            state: TaskState::Running,
            priority: TaskPriority::Normal,
            base_priority: TaskPriority::Normal,
            enqueued_at_tick: 0,
            exit_code: None,
            exit_waiters: crate::sched::WaitQueue::new(),
            is_user: true,
            wake_pending: false,
            pending_interrupt: false,
            affinity: crate::task::Affinity::Any,
            kstack_base: core::ptr::null_mut(),
            kstack_size: 0,
            kstack_top: 0,
            ctx: Default::default(),
            aspace: crate::sched::tests::MockAddressSpace(0),
            simd: crate::simd::SimdState::new(&crate::sched::tests::MOCK_RUNTIME),
            stack_info: None,
            mappings: alloc::sync::Arc::new(spin::Mutex::new(
                crate::memory::mappings::MappingList::new(),
            )),
            timeslice_remaining: crate::sched::types::DEFAULT_TIMESLICE,
            last_cpu: Some(0),
            name: [0; 32],
            name_len: 0,
            process_info: Some(pinfo),
            user_fs_base: 0,
            detached: false,
            signals: crate::signal::ThreadSignals::new(),
        };
        crate::task::registry::get_registry::<MockRuntime>().insert(alloc::boxed::Box::new(leader));
        sched.state.per_cpu[0].current = Some(leader_id);
    }

    /// A thread spawned with `tls_base = 0` must appear in the parent
    /// process's `thread_ids`.
    ///
    /// This directly tests the fix for the historical bug where the
    /// registration was inadvertently gated on `tls_base != 0`.
    #[test]
    fn test_spawn_thread_zero_tls_base_registered_in_thread_ids() {
        let _g = init_test_env();

        let leader_id: crate::task::TaskId = 7100;
        let pinfo = make_process_info(leader_id);

        let mut sched = Scheduler::<MockRuntime>::new();
        sched.next_id = 7101;
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        setup_leader_task(&mut sched, leader_id, pinfo.clone());

        let child_id = sched.spawn_user_thread(
            0x4000,
            0x8000,
            StartupArg::None,
            abi::types::StackInfo::default(),
            TaskPriority::Normal,
            crate::task::Affinity::Any,
            0, // tls_base = 0 — the historically broken case
            false,
        );

        let pi = pinfo.lock();
        assert!(
            pi.lifecycle.thread_ids.contains(&child_id),
            "thread spawned with tls_base=0 must appear in process thread_ids"
        );
        assert!(
            pi.lifecycle.thread_ids.contains(&leader_id),
            "leader TID must still be present after spawning a child"
        );
    }

    /// A thread spawned with a non-zero `tls_base` must also appear in the
    /// parent process's `thread_ids`.
    #[test]
    fn test_spawn_thread_nonzero_tls_base_registered_in_thread_ids() {
        let _g = init_test_env();

        let leader_id: crate::task::TaskId = 7200;
        let pinfo = make_process_info(leader_id);

        let mut sched = Scheduler::<MockRuntime>::new();
        sched.next_id = 7201;
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        setup_leader_task(&mut sched, leader_id, pinfo.clone());

        let child_id = sched.spawn_user_thread(
            0x4000,
            0x8000,
            StartupArg::None,
            abi::types::StackInfo::default(),
            TaskPriority::Normal,
            crate::task::Affinity::Any,
            0xDEAD_CAFE_0000_0000u64, // non-zero tls_base
            false,
        );

        let pi = pinfo.lock();
        assert!(
            pi.lifecycle.thread_ids.contains(&child_id),
            "thread spawned with non-zero tls_base must appear in process thread_ids"
        );
    }

    /// Calling `register_thread_in_process` twice for the same TID must not
    /// produce duplicate entries in `thread_ids`.
    #[test]
    fn test_register_thread_in_process_no_duplicates() {
        let _g = init_test_env();

        let pinfo = make_process_info(7300);
        let pinfo_opt = Some(pinfo.clone());

        // First registration (e.g. from spawn path).
        register_thread_in_process(&pinfo_opt, 7301);
        // Second registration (e.g. accidental double-call or re-use).
        register_thread_in_process(&pinfo_opt, 7301);

        let pi = pinfo.lock();
        let count = pi.lifecycle.thread_ids.iter().filter(|&&t| t == 7301).count();
        assert_eq!(count, 1, "duplicate TID entries must not be created");
    }

    /// Spawning multiple threads in sequence must register every one of them.
    #[test]
    fn test_multiple_threads_all_registered_in_thread_ids() {
        let _g = init_test_env();

        let leader_id: crate::task::TaskId = 7400;
        let pinfo = make_process_info(leader_id);

        let mut sched = Scheduler::<MockRuntime>::new();
        sched.next_id = 7401;
        sched.state.per_cpu.push(crate::sched::state::PerCpu::new());
        setup_leader_task(&mut sched, leader_id, pinfo.clone());

        let mut child_ids = alloc::vec::Vec::new();
        for _ in 0..4 {
            let id = sched.spawn_user_thread(
                0x4000,
                0x8000,
                StartupArg::None,
                abi::types::StackInfo::default(),
                TaskPriority::Normal,
                crate::task::Affinity::Any,
                0,
                false,
            );
            child_ids.push(id);
        }

        let pi = pinfo.lock();
        // Leader + 4 children = 5 entries, no duplicates.
        assert_eq!(
            pi.lifecycle.thread_ids.len(),
            5,
            "all spawned threads plus leader must be in thread_ids"
        );
        for &cid in &child_ids {
            assert!(
                pi.lifecycle.thread_ids.contains(&cid),
                "child TID {} must be in thread_ids",
                cid
            );
        }
    }

    // ── Runtime spawn path tests ──────────────────────────────────────────

    /// `spawn_process_from_path` must return `ENOENT` immediately when the
    /// requested path has no matching VFS mount.
    ///
    /// This verifies the first step of the runtime process creation model:
    /// "open executable" from the VFS.  If the path does not resolve, no
    /// scheduler or runtime interactions occur.
    #[test]
    fn test_spawn_process_from_path_returns_enoent_for_missing_vfs_path() {
        let _g = init_test_env();

        // Use a path that will not match any mount that might already be present.
        let result = unsafe {
            spawn_process_from_path::<MockRuntime>(
                "/totally/nonexistent/binary_9f3a1b",
                alloc::vec![],
                alloc::collections::BTreeMap::new(),
                StdioSpec::Inherit,
                StdioSpec::Inherit,
                StdioSpec::Inherit,
                0,
                alloc::vec![],
                None,
                alloc::vec![],
            )
        };

        assert!(
            matches!(result, Err(abi::errors::Errno::ENOENT)),
            "expected ENOENT for a path not present in the VFS, got {:?}",
            result
        );
    }

    /// `spawn_process_from_path` must return `EACCES` when the VFS node exists
    /// but is not a regular file (e.g. a directory).
    ///
    /// This verifies the runtime path's validation step before it ever touches
    /// the scheduler or ELF loader.
    #[test]
    fn test_spawn_process_from_path_returns_eacces_for_directory_node() {
        let _g = init_test_env();

        // A VFS node that pretends to be a directory.
        struct DirNode;
        impl crate::vfs::VfsNode for DirNode {
            fn read(&self, _: u64, _: &mut [u8]) -> abi::errors::SysResult<usize> {
                Ok(0)
            }
            fn write(&self, _: u64, _: &[u8]) -> abi::errors::SysResult<usize> {
                Ok(0)
            }
            fn stat(&self) -> abi::errors::SysResult<crate::vfs::VfsStat> {
                Ok(crate::vfs::VfsStat {
                    mode: crate::vfs::VfsStat::S_IFDIR | 0o755,
                    size: 0,
                    ino: 1,
                    ..Default::default()
                })
            }
        }

        struct DirFs;
        impl crate::vfs::VfsDriver for DirFs {
            fn lookup(
                &self,
                path: &str,
            ) -> abi::errors::SysResult<alloc::sync::Arc<dyn crate::vfs::VfsNode>> {
                if path == "notafile" {
                    Ok(alloc::sync::Arc::new(DirNode))
                } else {
                    Err(abi::errors::Errno::ENOENT)
                }
            }
        }

        crate::vfs::mount::init();
        crate::vfs::mount::mount("/spawn_test_dir", alloc::sync::Arc::new(DirFs));

        let result = unsafe {
            spawn_process_from_path::<MockRuntime>(
                "/spawn_test_dir/notafile",
                alloc::vec![],
                alloc::collections::BTreeMap::new(),
                StdioSpec::Inherit,
                StdioSpec::Inherit,
                StdioSpec::Inherit,
                0,
                alloc::vec![],
                None,
                alloc::vec![],
            )
        };

        let _ = crate::vfs::mount::umount("/spawn_test_dir");

        assert!(
            matches!(result, Err(abi::errors::Errno::EACCES)),
            "expected EACCES for a directory node, got {:?}",
            result
        );
    }

    /// The runtime hook `SPAWN_PROCESS_FROM_PATH_HOOK` is a separate static
    /// from the boot hook `SPAWN_PROCESS_EX_HOOK`.
    ///
    /// Before the scheduler is initialized the hook is `None`, so calling
    /// `spawn_process_from_path_current` returns `ENOSYS`.  This verifies
    /// the hook plumbing and its independence from the boot path.
    #[test]
    fn test_spawn_process_from_path_current_returns_enosys_without_scheduler() {
        let _g = init_test_env();
        // `init_test_env` sets SCHEDULER to None, so hooks are not installed.
        let result = unsafe {
            crate::sched::hooks::spawn_process_from_path_current(
                "/usr/bin/ls",
                alloc::vec![],
                alloc::collections::BTreeMap::new(),
                StdioSpec::Inherit,
                StdioSpec::Inherit,
                StdioSpec::Inherit,
                0,
                alloc::vec![],
                None,
                alloc::vec![],
            )
        };

        assert!(
            matches!(result, Err(abi::errors::Errno::ENOSYS)),
            "expected ENOSYS when SPAWN_PROCESS_FROM_PATH_HOOK is not installed, got {:?}",
            result
        );
    }
}
