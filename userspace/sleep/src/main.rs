#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::string::String;
use alloc::vec::Vec;
use stem::syscall::{argv_get, exit, sleep_ns, vfs_write};

fn get_args() -> Vec<String> {
    let mut len = 0;
    if let Ok(l) = argv_get(&mut []) {
        len = l;
    }
    if len == 0 {
        return Vec::new();
    }

    let mut buf = alloc::vec![0u8; len];
    if argv_get(&mut buf).is_err() {
        return Vec::new();
    }

    stem::utils::parse_argv(&buf)
        .into_iter()
        .skip(1)
        .filter_map(|b| core::str::from_utf8(b).ok().map(String::from))
        .collect()
}

fn print_err(msg: &str) {
    let _ = vfs_write(2, msg.as_bytes());
}

/// Parse a duration string like "1", "1.5", "1s", "500ms", "2m", "1h".
/// Returns duration in nanoseconds, or None on parse failure.
fn parse_duration_ns(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Detect suffix
    let (num_str, multiplier_ns): (&str, u64) = if s.ends_with("ms") {
        (&s[..s.len() - 2], 1_000_000)
    } else if s.ends_with('s') {
        (&s[..s.len() - 1], 1_000_000_000)
    } else if s.ends_with('m') {
        (&s[..s.len() - 1], 60_000_000_000)
    } else if s.ends_with('h') {
        (&s[..s.len() - 1], 3_600_000_000_000)
    } else {
        (s, 1_000_000_000)
    };

    // Try integer parse first
    if let Ok(n) = num_str.parse::<u64>() {
        return Some(n.saturating_mul(multiplier_ns));
    }

    // Try floating point parse
    if let Some(dot) = num_str.find('.') {
        let int_part: u64 = if dot == 0 {
            0
        } else {
            num_str[..dot].parse().ok()?
        };
        let frac_str = &num_str[dot + 1..];
        if frac_str.is_empty() {
            return Some(int_part.saturating_mul(multiplier_ns));
        }
        // Parse fractional part: limit to 9 decimal digits (nanosecond precision)
        let frac_digits = frac_str.len().min(9);
        let frac_val: u64 = frac_str[..frac_digits].parse().ok()?;
        // Scale frac_val to nanoseconds
        let scale = 10u64.pow((9 - frac_digits) as u32);
        let frac_ns = frac_val.saturating_mul(scale);
        // frac_ns is the fractional part scaled to nanoseconds (for one unit).
        // Convert fractional ns to the actual suffix unit via:
        //   frac_real_ns = frac_ns * multiplier_ns / 1_000_000_000
        // Use u128 to avoid overflow / premature truncation (e.g. for "ms" suffix).
        let frac_real_ns =
            (frac_ns as u128 * multiplier_ns as u128 / 1_000_000_000u128) as u64;
        let total_ns = int_part
            .saturating_mul(multiplier_ns)
            .saturating_add(frac_real_ns);
        return Some(total_ns);
    }

    None
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.is_empty() {
        print_err("usage: sleep <duration>[suffix]...\n");
        print_err("suffixes: s (seconds, default), ms (milliseconds), m (minutes), h (hours)\n");
        exit(1)
    }

    let mut total_ns: u64 = 0;
    for arg in &args {
        match parse_duration_ns(arg) {
            Some(ns) => total_ns = total_ns.saturating_add(ns),
            None => {
                let msg = alloc::format!("sleep: invalid time interval '{}'\n", arg);
                print_err(&msg);
                exit(1)
            }
        }
    }

    sleep_ns(total_ns);
    exit(0)
}
