//! Smoke test: Instant (monotonic) and SystemTime (wall clock) via syscalls.
//!
//! Acceptance criteria:
//!   - `Instant::now()` returns non-zero values
//!   - Two successive `Instant::now()` calls are non-decreasing
//!   - `Instant::elapsed()` returns a non-negative duration
//!   - `SystemTime::now()` returns a value >= UNIX_EPOCH (once RTC is set)
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use std::time::{Duration, Instant, SystemTime};

fn main() {
    // ── Instant (monotonic) ──────────────────────────────────────────────────
    let t0 = Instant::now();
    println!("[time_test] t0 = {:?}", t0);

    // Busy-wait for a small amount
    let mut sum: u64 = 0;
    for i in 0..100_000u64 {
        sum = sum.wrapping_add(i);
    }
    // Prevent the loop from being optimized away
    if sum == 0 {
        println!("[time_test] (sum=0, unexpected)");
    }

    let t1 = Instant::now();
    println!("[time_test] t1 = {:?}", t1);

    match t1.checked_duration_since(t0) {
        Some(d) => println!("[time_test] elapsed = {:?}", d),
        None => {
            eprintln!("[time_test] FAIL: t1 < t0 (time went backwards)");
            std::process::exit(1);
        }
    }

    // elapsed() helper
    let elapsed = t0.elapsed();
    println!("[time_test] t0.elapsed() = {:?}", elapsed);
    if elapsed < Duration::from_nanos(1) {
        eprintln!("[time_test] WARN: elapsed < 1ns — clock may be stuck");
    }

    // Duration arithmetic
    let d = Duration::from_millis(50);
    let t2 = t0 + d;
    let t3 = t2 - d;
    if t3 != t0 {
        eprintln!("[time_test] FAIL: duration add/sub roundtrip failed");
        std::process::exit(1);
    }
    println!("[time_test] Instant arithmetic OK");

    // ── SystemTime (wall clock) ───────────────────────────────────────────────
    let now = SystemTime::now();
    println!("[time_test] SystemTime::now() = {:?}", now);

    match now.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(since_epoch) => {
            println!("[time_test] seconds since UNIX_EPOCH = {}", since_epoch.as_secs());
            if since_epoch.is_zero() {
                println!("[time_test] WARN: wall clock at epoch (RTC not set?)");
            }
        }
        Err(e) => {
            // Before epoch is unusual but not fatal (RTC may be unset/zero)
            println!("[time_test] WARN: SystemTime before UNIX_EPOCH by {:?}", e.duration());
        }
    }

    // Two SystemTime readings should be non-decreasing
    let st0 = SystemTime::now();
    let st1 = SystemTime::now();
    if st1 < st0 {
        eprintln!("[time_test] FAIL: SystemTime went backwards");
        std::process::exit(1);
    }
    println!("[time_test] SystemTime monotonicity OK");

    println!("[time_test] PASS");
    std::process::exit(0);
}
