# Kernel

The **Kernel** (`/kernel`) is the core logic of ThingOS. It is architecture-agnostic, relying on **Bran** for hardware abstraction during boot.

## Role

The kernel's primary responsibility is to host the **Root** service (the graph database) and provide the environment for userspace processes (like **Sprout**) to run.

Unlike traditional monolithic kernels, ThingOS kernel is relatively minimal. It provides:

*   **Memory Management**: Paging, allocation (building on Bran's memory map).
*   **Task Scheduling**: Preemptive multitasking for kernel threads and userspace processes.
*   **IPC**: Inter-Process Communication, primarily via the Graph (Root) and direct messaging.
*   **Root Hosting**: It runs the Root service as a high-priority kernel thread.

## Relationship with Bran

The kernel receives a `BootRuntime` trait object from Bran at entry. It uses this to interact with the hardware (console, memory map, etc.) without knowing the underlying architecture details.

```rust
pub fn start<R: BootRuntime>(runtime: &'static R) -> ! {
    // ...
}
```

## Startup Sequence

1.  **Entry**: Bran calls `kernel::start(runtime)`.
2.  **Init**: Kernel initializes memory, logging, and tasking.
3.  **Root Spawn**: The kernel spawns the `Root` service thread (`crate::root::init_root_service`).
4.  **Registration**: The kernel registers itself and the hardware inventory into the Root graph.
5.  **Sprout Load**: The kernel looks for the `sprout` boot module (the init process).
6.  **Userland Launch**: The kernel creates a user address space and spawns `sprout`.
7.  **Idle Loop**: The main kernel thread enters the scheduler/idle loop.

## Key Subsystems

*   `root/`: The implementation of the graph database.
*   `task/`: Scheduler and process management.
*   `memory/`: Virtual and physical memory management.
*   `syscall/`: System call handlers (interface for **Stem**).

## Thing Ownership and Lifecycle

ThingOS implements automatic resource management for things (nodes in the graph) through ownership tracking:

### Ownership Model

- **Each thing can have an owner**: When a process creates a thing, it becomes the owner of that thing.
- **Owner tracking**: The graph maintains an `owner` field in each `Node` that stores the ThingId of the owning process's graph node.
- **Kernel-owned things**: Things created during boot or explicitly orphaned have `owner = None` (kernel-owned).

### Automatic Cleanup

When a process terminates (either via `exit()` or being killed):
1. The scheduler identifies the process's graph thing ID
2. All things owned by that process are automatically destroyed via `CleanupTaskThings` operation
3. This prevents resource leaks and ensures cleanup happens even on abnormal termination

### Orphaning Things

Processes can explicitly transfer ownership to the kernel using the `SYS_ROOT_ORPHAN_THING` syscall:
```rust
stem::syscall::graph::orphan_thing(thing_id)?;
```

This is useful for:
- Creating long-lived resources that should outlive the creating process
- Implementing daemon services that manage shared resources
- Transferring ownership to the kernel before process exit

### Implementation Details

- **Owner field**: Added to `Node` structure in `kernel/src/root/graph.rs`
- **Syscall**: `SYS_ROOT_ORPHAN_THING` (0x173)
- **Cleanup**: Triggered in `terminate_current()` and `kill_by_tid()` in the scheduler
- **Graph operations**: `get_owned_things()`, `orphan_thing()`, `set_owner()` methods on `Graph`
