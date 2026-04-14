use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

/// Unique identifier for a kernel thread (scheduler task).
pub type ThreadId = u64;
/// Backward-compatible alias — prefer `ThreadId` in new code.
pub type TaskId = ThreadId;

/// Scheduling priority levels; higher variants preempt lower ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreadPriority {
    Idle = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Realtime = 4,
}
/// Backward-compatible alias — prefer `ThreadPriority` in new code.
pub type TaskPriority = ThreadPriority;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Affinity {
    Any,
    Pinned(usize),
}

/// Lifecycle state of a kernel thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    Runnable,
    Running,
    Blocked,
    Dead,
}
/// Backward-compatible alias — prefer `ThreadState` in new code.
pub type TaskState = ThreadState;

/// Scheduler-side metadata for a single kernel thread.
///
/// This is the **single source of truth** for scheduler-internal queue
/// tracking.  All other scheduling fields (`state`, `priority`,
/// `base_priority`, `timeslice_remaining`, `affinity`, `enqueued_at_tick`,
/// `last_cpu`) live exclusively in `Thread<R>` (the registry entry).
///
/// `ThreadSchedFields` only carries the two fields that have no equivalent
/// in `Thread<R>`:
/// - `tid` — needed to keep the scheduler's sorted `Vec` indexed in the same
///   order as `ThreadRegistry::threads` so that a single binary-search index
///   addresses both collections.
/// - `runq_location` — tracks which `(cpu, priority)` run-queue slot currently
///   holds this thread; there is no corresponding field in `Thread<R>`.
pub struct ThreadSchedFields {
    pub tid: ThreadId,
    pub runq_location: Option<(usize, usize)>,
}
/// Backward-compatible alias — prefer `ThreadSchedFields` in new code.
pub type TaskSchedFields = ThreadSchedFields;

#[derive(Debug, Clone, Copy)]
pub struct SleepEntry {
    pub tid: ThreadId,
    pub wake_tick: u64,
}

pub struct PerCpu {
    pub runq: [VecDeque<ThreadId>; 5],
    pub idle_task: Option<ThreadId>,
    pub current: Option<ThreadId>,
    pub last_switch: u64,
    pub need_resched: bool,
}

impl PerCpu {
    pub fn new() -> Self {
        PerCpu {
            runq: [
                VecDeque::with_capacity(128),
                VecDeque::with_capacity(128),
                VecDeque::with_capacity(128),
                VecDeque::with_capacity(128),
                VecDeque::with_capacity(128),
            ],
            idle_task: None,
            current: None,
            last_switch: 0,
            need_resched: false,
        }
    }
}

pub struct SchedState {
    pub threads: Vec<ThreadSchedFields>,
    pub per_cpu: Vec<PerCpu>,
    pub sleep_queue: BTreeMap<u64, Vec<ThreadId>>,
    pub wait_queue: VecDeque<ThreadId>,
    pub online_cpu_count: usize,
    pub online_cpus: Vec<usize>,
}

impl SchedState {
    pub fn new() -> Self {
        SchedState {
            threads: Vec::with_capacity(1024),
            per_cpu: Vec::with_capacity(32),
            sleep_queue: BTreeMap::new(),
            wait_queue: VecDeque::with_capacity(1024),
            online_cpu_count: 1,
            online_cpus: Vec::with_capacity(32),
        }
    }

    pub fn set_boot_cpu_online(&mut self) {
        self.online_cpus.clear();
        self.online_cpus.push(0);
        self.online_cpu_count = 1;
    }

    pub fn mark_cpu_online(&mut self, cpu: usize) {
        if !self.online_cpus.contains(&cpu) {
            self.online_cpus.push(cpu);
            self.online_cpus.sort_unstable();
        }
        self.online_cpu_count = self.online_cpus.len();
    }

    pub fn pick_online_cpu_excluding_bsp(&self, rr_idx: usize) -> usize {
        if self.online_cpus.len() <= 1 {
            return 0;
        }
        let secondary = &self.online_cpus[1..];
        secondary[rr_idx % secondary.len()]
    }

    pub fn pick_online_cpu(&self, rr_idx: usize) -> usize {
        if self.online_cpus.is_empty() {
            0
        } else {
            self.online_cpus[rr_idx % self.online_cpus.len()]
        }
    }

    pub fn get_thread_index(&self, tid: ThreadId) -> Option<usize> {
        self.threads.binary_search_by_key(&tid, |t| t.tid).ok()
    }

    pub fn get_thread(&self, tid: ThreadId) -> Option<&ThreadSchedFields> {
        self.get_thread_index(tid).map(|idx| &self.threads[idx])
    }

    pub fn get_thread_mut(&mut self, tid: ThreadId) -> Option<&mut ThreadSchedFields> {
        self.get_thread_index(tid)
            .map(move |idx| &mut self.threads[idx])
    }

    pub fn insert_thread(&mut self, fields: ThreadSchedFields) {
        match self.threads.binary_search_by_key(&fields.tid, |t| t.tid) {
            Ok(_) => panic!("Thread ID {} already exists in sched", fields.tid),
            Err(idx) => self.threads.insert(idx, fields),
        }
    }

    pub fn enqueue_thread(&mut self, cpu: usize, prio: usize, tid: ThreadId) {
        if let Some(pc) = self.per_cpu.get_mut(cpu) {
            pc.runq[prio].push_back(tid);
        }
        if let Some(t) = self.get_thread_mut(tid) {
            t.runq_location = Some((cpu, prio));
        }
    }

    pub fn dequeue_thread_front(&mut self, cpu: usize, prio: usize) -> Option<ThreadId> {
        if let Some(pc) = self.per_cpu.get_mut(cpu) {
            if let Some(tid) = pc.runq[prio].pop_front() {
                if let Some(t) = self.get_thread_mut(tid) {
                    t.runq_location = None;
                }
                return Some(tid);
            }
        }
        None
    }

    pub fn remove_thread_from_runq(&mut self, tid: ThreadId) -> bool {
        let (cpu, prio) = match self.get_thread(tid) {
            Some(t) if t.runq_location.is_some() => t.runq_location.unwrap(),
            _ => return false,
        };

        if let Some(pc) = self.per_cpu.get_mut(cpu) {
            if let Some(pos) = pc.runq[prio].iter().position(|&id| id == tid) {
                pc.runq[prio].remove(pos);
                if let Some(t) = self.get_thread_mut(tid) {
                    t.runq_location = None;
                }
                return true;
            }
        }
        false
    }

    pub fn remove_thread(&mut self, tid: ThreadId) -> bool {
        if let Ok(idx) = self.threads.binary_search_by_key(&tid, |t| t.tid) {
            self.threads.remove(idx);
            true
        } else {
            false
        }
    }

    // ── Backward-compatible forwarding methods ────────────────────────────────

    #[inline]
    pub fn get_task_index(&self, tid: ThreadId) -> Option<usize> {
        self.get_thread_index(tid)
    }
    #[inline]
    pub fn get_task(&self, tid: ThreadId) -> Option<&ThreadSchedFields> {
        self.get_thread(tid)
    }
    #[inline]
    pub fn get_task_mut(&mut self, tid: ThreadId) -> Option<&mut ThreadSchedFields> {
        self.get_thread_mut(tid)
    }
    #[inline]
    pub fn insert_task(&mut self, fields: ThreadSchedFields) {
        self.insert_thread(fields)
    }
    #[inline]
    pub fn enqueue_task(&mut self, cpu: usize, prio: usize, tid: ThreadId) {
        self.enqueue_thread(cpu, prio, tid)
    }
    #[inline]
    pub fn dequeue_task_front(&mut self, cpu: usize, prio: usize) -> Option<ThreadId> {
        self.dequeue_thread_front(cpu, prio)
    }
    #[inline]
    pub fn remove_task_from_runq(&mut self, tid: ThreadId) -> bool {
        self.remove_thread_from_runq(tid)
    }
    #[inline]
    pub fn remove_task(&mut self, tid: ThreadId) -> bool {
        self.remove_thread(tid)
    }
}
