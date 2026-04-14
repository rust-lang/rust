# Driver Syscalls v0

This document describes the minimal syscall surface for early drivers and leaves in ThingOS v0.

## Principles

1.  **Kernel Ownership**: Kernel owns hardware privileges (ports, MMIO, interrupts).
2.  **Safety**: Syscalls must be safe under cooperative scheduling and eventual preemption. Pointers are validated.
3.  **Error Handling**: Returns negative errno-style integers (via registers).

## Syscalls

### 1. Logging

**`sys_log_write`** (SYS_LOG_WRITE = 11)

Writes a string to the kernel log sink (serial/framebuffer).

*   **Args**:
    *   `ptr`: `*const u8` (pointer to buffer)
    *   `len`: `usize` (length of buffer)
*   **Returns**: `isize` (bytes written or error)
*   **Errors**:
    *   `EFAULT`: Invalid memory range.

### 2. Time

**`sys_time_monotonic_ns`** (SYS_TIME_MONOTONIC = 8)

Returns monotonic time in nanoseconds since boot.

*   **Args**: None
*   **Returns**: `isize` (casted `u64` nanoseconds).
    *   *Note*: In v0, we assume uptime < 292 years so positive signed integer holds the value.

**`sys_rtc_read`** (SYS_RTC_READ = 9)

Reads the Real Time Clock (CMOS) and fills the provided struct.

*   **Args**:
    *   `out_ptr`: `*mut RtcTime`
*   **Returns**: `isize` (0 on success)
*   **Errors**:
    *   `EFAULT`: Invalid memory range.
    *   `ENODEV`: RTC not supported/present.

**Struct `RtcTime`**:
```rust
#[repr(C)]
pub struct RtcTime {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub weekday: u8,
    pub flags: u8,
}
```

### 3. Scheduling

**`sys_yield`** (SYS_YIELD = 5)

Yields execution to the scheduler (cooperative).

*   **Args**: None
*   **Returns**: `isize` (0)

**`sys_sleep_ns`** (SYS_SLEEP_NS = 10)

Sleeps until `now + ns`.

*   **Args**:
    *   `ns`: `u64` (nanoseconds to sleep)
*   **Returns**: `isize` (0)

### 4. Process/Thread

**`sys_exit`** (SYS_EXIT = 1)

Terminates the current task.

*   **Args**:
    *   `code`: `i32`
*   **Returns**: Does not return.

**`sys_get_tid`** (SYS_GET_TID = 12)

Returns the current thread ID.

*   **Args**: None
*   **Returns**: `isize` (Thread ID).
