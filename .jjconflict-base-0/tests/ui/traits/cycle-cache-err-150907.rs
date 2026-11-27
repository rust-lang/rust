// Test that we don't hang or take exponential time when evaluating
// auto-traits for cyclic types containing errors, based on the reproducer
// provided in issue #150907.

use std::sync::Arc;
use std::marker::PhantomData;

struct Weak<T>(PhantomData<T>);
unsafe impl<T: Sync + Send> Send for Weak<T> {}
unsafe impl<T: Sync + Send> Sync for Weak<T> {}

struct BTreeMap<K, V>(K, V);

trait DeviceOps: Send {}

struct SerialDevice {
    terminal: Weak<Terminal>,
}

impl DeviceOps for SerialDevice {}

struct TtyState {
    terminals: Weak<Terminal>,
}

struct TerminalMutableState {
    controller: TerminalController,
}

struct Terminal {
    weak_self: Weak<Self>,
    state: Arc<TtyState>,
    mutable_state: Weak<TerminalMutableState>,
}

struct TerminalController {
    session: Weak<Session>,
}

struct Kernel {
    weak_self: Weak<Kernel>,
    kthreads: KernelThreads,
    pids: Weak<PidTable>,
}

struct KernelThreads {
    system_task: SystemTask,
    kernel: Weak<Kernel>,
}

struct SystemTask {
    system_thread_group: Weak<ThreadGroup>,
}

enum ProcessEntry {
    ThreadGroup(Weak<ThreadGroup>),
}

struct PidEntry {
    task: Arc<Task>,
    process: ProcessEntry,
}

struct PidTable {
    table: PidEntry,
    process_groups: Arc<ProcessGroup>,
}

struct ProcessGroupMutableState {
    thread_groups: Weak<ThreadGroup>,
}

struct ProcessGroup {
    session: Arc<Session>,
    mutable_state: Arc<ProcessGroupMutableState>,
}

struct SessionMutableState {
    process_groups: BTreeMap<(), Weak<ProcessGroup>>,
    controlling_terminal: ControllingTerminal,
}

struct Session {
    mutable_state: Weak<SessionMutableState>,
}

struct ControllingTerminal {
    terminal: Arc<Terminal>,
}

struct TaskPersistentInfoState {
    thread_group_key: ThreadGroupKey,
}

struct Task {
    thread_group_key: ThreadGroupKey,
    kernel: Arc<Kernel>,
    thread_group: Arc<ThreadGroup>,
    persistent_info: Arc<TaskPersistentInfoState>,
    vfork_event: Arc, //~ ERROR missing generics for struct `Arc`
}

struct ThreadGroupKey {
    thread_group: Arc<ThreadGroup>,
}

struct ThreadGroupMutableState {
    tasks: BTreeMap<(), TaskContainer>,
    children: BTreeMap<(), Weak<ThreadGroup>>,
    process_group: Arc<ProcessGroup>,
}

struct ThreadGroup {
    kernel: Arc<Kernel>,
    mutable_state: Weak<ThreadGroupMutableState>,
}

struct TaskContainer(Arc<Task>, Arc<TaskPersistentInfoState>);

fn main() {
    // Trigger auto-trait check for one of the cyclic types
    is_send::<Kernel>();
}

fn is_send<T: Send>() {}
