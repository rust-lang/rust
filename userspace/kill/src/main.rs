//! `kill` — send signals to processes.
//!
//! Usage: kill [-<signal>] <pid> [<pid> ...]
//!        kill -l
//!
//! With no signal specified, sends SIGTERM (15).

#![no_std]
#![no_main]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use abi::signal::*;
use stem::syscall::{argv_get, kill, vfs_write};

fn write_stderr(msg: &str) {
    let _ = unsafe { vfs_write(2, msg.as_bytes()) };
}

fn write_stdout(msg: &str) {
    let _ = unsafe { vfs_write(1, msg.as_bytes()) };
}

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
    let mut args = Vec::new();
    if buf.len() >= 4 {
        let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        for _ in 0..count {
            if offset + 4 > buf.len() {
                break;
            }
            let slen = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + slen > buf.len() {
                break;
            }
            let s = String::from_utf8_lossy(&buf[offset..offset + slen]).into_owned();
            args.push(s);
            offset += slen;
        }
    }
    args
}

/// Parse a signal name (e.g. "TERM", "SIGTERM", "15") to its number.
fn parse_signal(s: &str) -> Option<u8> {
    // Strip optional "SIG" prefix.
    let s = s.strip_prefix("SIG").unwrap_or(s);
    // Try as a decimal number first.
    if let Ok(n) = s.parse::<u8>() {
        if n < NSIG {
            return Some(n);
        }
        return None;
    }
    // Match by name.
    match s {
        "HUP" => Some(SIGHUP),
        "INT" => Some(SIGINT),
        "QUIT" => Some(SIGQUIT),
        "ILL" => Some(SIGILL),
        "TRAP" => Some(SIGTRAP),
        "ABRT" | "IOT" => Some(SIGABRT),
        "BUS" => Some(SIGBUS),
        "FPE" => Some(SIGFPE),
        "KILL" => Some(SIGKILL),
        "USR1" => Some(SIGUSR1),
        "SEGV" => Some(SIGSEGV),
        "USR2" => Some(SIGUSR2),
        "PIPE" => Some(SIGPIPE),
        "ALRM" => Some(SIGALRM),
        "TERM" => Some(SIGTERM),
        "CHLD" | "CLD" => Some(SIGCHLD),
        "CONT" => Some(SIGCONT),
        "STOP" => Some(SIGSTOP),
        "TSTP" => Some(SIGTSTP),
        "TTIN" => Some(SIGTTIN),
        "TTOU" => Some(SIGTTOU),
        "URG" => Some(SIGURG),
        "XCPU" => Some(SIGXCPU),
        "XFSZ" => Some(SIGXFSZ),
        "VTALRM" => Some(SIGVTALRM),
        "PROF" => Some(SIGPROF),
        "WINCH" => Some(SIGWINCH),
        "IO" | "POLL" => Some(SIGIO),
        "PWR" => Some(SIGPWR),
        "SYS" => Some(SIGSYS),
        _ => None,
    }
}

fn sig_name(sig: u8) -> &'static str {
    match sig {
        SIGHUP => "HUP",
        SIGINT => "INT",
        SIGQUIT => "QUIT",
        SIGILL => "ILL",
        SIGTRAP => "TRAP",
        SIGABRT => "ABRT",
        SIGBUS => "BUS",
        SIGFPE => "FPE",
        SIGKILL => "KILL",
        SIGUSR1 => "USR1",
        SIGSEGV => "SEGV",
        SIGUSR2 => "USR2",
        SIGPIPE => "PIPE",
        SIGALRM => "ALRM",
        SIGTERM => "TERM",
        SIGSTKFLT => "STKFLT",
        SIGCHLD => "CHLD",
        SIGCONT => "CONT",
        SIGSTOP => "STOP",
        SIGTSTP => "TSTP",
        SIGTTIN => "TTIN",
        SIGTTOU => "TTOU",
        SIGURG => "URG",
        SIGXCPU => "XCPU",
        SIGXFSZ => "XFSZ",
        SIGVTALRM => "VTALRM",
        SIGPROF => "PROF",
        SIGWINCH => "WINCH",
        SIGIO => "IO",
        SIGPWR => "PWR",
        SIGSYS => "SYS",
        _ => "?",
    }
}

fn print_signal_list() {
    for n in 1u8..SIGRTMIN {
        let name = sig_name(n);
        if name != "?" {
            let mut line = alloc::format!("{n}) SIG{name}\n");
            write_stdout(&line);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    let args = get_args();

    // Skip argv[0] (the program name).
    let args: Vec<&str> = args.iter().skip(1).map(|s| s.as_str()).collect();

    if args.is_empty() {
        write_stderr("Usage: kill [-<signal>] <pid> [<pid> ...]\n");
        write_stderr("       kill -l\n");
        return 1;
    }

    // Handle `kill -l`.
    if args[0] == "-l" || args[0] == "--list" {
        print_signal_list();
        return 0;
    }

    // Parse optional signal argument (e.g. -9, -SIGTERM, -TERM).
    let (sig, pids_start) = if let Some(sig_str) = args[0].strip_prefix('-') {
        match parse_signal(sig_str) {
            Some(s) => (s, 1),
            None => {
                let msg = alloc::format!("kill: unknown signal: -{sig_str}\n");
                write_stderr(&msg);
                return 1;
            }
        }
    } else {
        (SIGTERM, 0)
    };

    if pids_start >= args.len() {
        write_stderr("kill: no PIDs specified\n");
        return 1;
    }

    let mut exit_code = 0i32;
    for pid_str in &args[pids_start..] {
        match pid_str.parse::<i32>() {
            Ok(pid) => match kill(pid, sig) {
                Ok(()) => {}
                Err(e) => {
                    let msg = alloc::format!("kill: {pid}: {e:?}\n");
                    write_stderr(&msg);
                    exit_code = 1;
                }
            },
            Err(_) => {
                let msg = alloc::format!("kill: invalid PID: {pid_str}\n");
                write_stderr(&msg);
                exit_code = 1;
            }
        }
    }

    exit_code
}
