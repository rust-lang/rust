# Scheduler Anti-Starvation Mechanism

## Problem

Thing-OS uses a priority-based preemptive scheduler with 5 priority levels:
- Idle (0)
- Low (1) 
- Normal (2)
- High (3)
- Realtime (4)

The scheduler always picks the highest-priority runnable task. While this ensures high-priority work gets done quickly, it can lead to **starvation**: low-priority tasks may never get CPU time if high-priority tasks are continuously runnable.

## Solution: Priority Aging

Priority aging is a common anti-starvation technique where tasks that have been waiting too long receive a temporary priority boost. Thing-OS implements this through three mechanisms:

### 1. Wait Time Tracking

Each task has two new fields:
- `wait_ticks: u64` - Counts ticks since the task last ran
- `base_priority: TaskPriority` - The task's original priority

On each timer tick (100Hz, so every ~10ms), the scheduler increments `wait_ticks` for all runnable tasks except the currently running one.

### 2. Priority Aging

Also on each timer tick, the scheduler applies aging:

```rust
boost_levels = wait_ticks / AGING_THRESHOLD_TICKS
boost_levels = min(boost_levels, MAX_PRIORITY_BOOST)
effective_priority = min(base_priority + boost_levels, Realtime)
```

**Constants:**
- `AGING_THRESHOLD_TICKS = 500` (~5 seconds at 100Hz)
- `MAX_PRIORITY_BOOST = 2` (maximum 2 priority levels boost)

**Example:**
- A Low-priority task waits for 1000 ticks (10 seconds)
- Boost = 1000 / 500 = 2 levels
- Effective priority = Low + 2 = High
- The task now competes in the High priority queue

### 3. Reset on Schedule

When a task is finally scheduled to run:
1. `wait_ticks` is reset to 0
2. `priority` is restored to `base_priority`

This ensures the boost is temporary and only applies while the task is starving.

## Configuration

The aging behavior can be tuned by changing the constants in `kernel/src/task/scheduler/types.rs`:

- **AGING_THRESHOLD_TICKS**: How long before first boost. Lower values = more aggressive anti-starvation
- **MAX_PRIORITY_BOOST**: Maximum boost levels. Higher values allow more aggressive boosting

## Trade-offs

**Benefits:**
- Prevents indefinite starvation of low-priority tasks
- Low overhead (simple arithmetic on timer tick)
- Predictable behavior based on wait time

**Drawbacks:**
- Boosted low-priority tasks can temporarily delay high-priority work
- Adds small overhead to every timer tick
- Two additional 64-bit fields per task (16 bytes)

## Implementation Details

The mechanism is implemented in three methods in `kernel/src/task/scheduler/mod.rs`:

1. **`increment_wait_times(cpu_idx)`** - Increments wait_ticks for waiting tasks
2. **`apply_priority_aging(cpu_idx)`** - Calculates and applies priority boosts
3. **`reset_wait_time(task_id)`** - Resets state when task is scheduled

These are called from:
- `schedule_point()` with `ScheduleReason::PreemptTick` (aging and increment)
- `prepare_schedule()` when switching to a new task (reset)

## Testing

Unit tests in `kernel/src/task/scheduler/mod.rs` verify:
- `test_wait_ticks_increment` - Wait times increment correctly
- `test_priority_aging_boost` - Priority boosts after aging threshold
- `test_reset_wait_time_on_schedule` - State resets when scheduled

Run tests with:
```bash
cargo test --package kernel --lib scheduler::tests
```

## Future Improvements

Possible enhancements:
1. Per-priority-level aging thresholds (e.g., Low tasks age faster than Normal)
2. Adaptive aging based on system load
3. Statistics tracking (how often aging triggers, average wait times)
4. User-configurable aging policies via syscall
