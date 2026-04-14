# Scheduling Model (v0)

## Current State: Cooperative Round-Robin

The scheduler uses a simple round-robin policy with a single run queue (`VecDeque`).

### Key Structures
- **Scheduler**: Protected by a global `spin::Mutex`.
- **Run Queue**: `VecDeque<TaskId>` of runnable tasks.
- **Task List**: `Vec<Task>` storage for all tasks.

### Context Switching
Context switching is triggered explicitly via `yield_now`.
1. **Interrupts Disabled**: To prevent race conditions during switch.
2. **Lock Acquired**: The global scheduler lock is taken.
3. **Prepare Yield**:
   - Current task is pushed to back of run queue.
   - Next task is popped from front.
   - Task states updated (`Running` <-> `Runnable`).
   - SIMD state saved/restored.
4. **Switch**: `ArchRuntime::switch` is called to swap stack pointers and registers.
5. **Lock Released**: Implicitly (via scope drop, though currently `yield_now` logic handles pointers carefully).
6. **Interrupts Restored**: After returning from `switch`.

### Constraints
- **Deeply Coupled**: `Scheduler` logic is somewhat intertwined with `yield_now` implementation in `scheduler.rs`.
- **No Preemption**: Timer ticks do not currently trigger rescheduling.
