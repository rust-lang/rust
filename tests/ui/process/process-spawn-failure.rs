//! Tests that repeatedly spawning a failing command does not create zombie processes.
//! Spawns a deliberately invalid command multiple times, verifies each spawn fails,
//! then uses `ps` (on Unix) to detect any leftover zombie (defunct) child processes.
//! Checks Rust's process spawning cleans up resources properly.
//! Skipped on platforms without `ps` utility.

//@ run-pass
//@ needs-subprocess
//@ ignore-vxworks no 'ps'
//@ ignore-fuchsia no 'ps'
//@ ignore-nto no 'ps'
//@ ignore-ios no 'ps'
//@ ignore-tvos no 'ps'
//@ ignore-watchos no 'ps'
//@ ignore-visionos no 'ps'

#![feature(rustc_private)]

use std::process::Command;

// The output from "ps -A -o pid,ppid,args" should look like this:
//   PID  PPID COMMAND
//     1     0 /sbin/init
//     2     0 [kthreadd]
// ...
//  6076  9064 /bin/zsh
// ...
//  7164  6076 ./spawn-failure
//  7165  7164 [spawn-failure] <defunct>
//  7166  7164 [spawn-failure] <defunct>
// ...
//  7197  7164 [spawn-failure] <defunct>
//  7198  7164 ps -A -o pid,ppid,command
// ...

#[cfg(unix)]
fn find_zombies() {
    extern crate libc;
    let my_pid = unsafe { libc::getpid() };

    // https://pubs.opengroup.org/onlinepubs/9699919799/utilities/ps.html
    let ps_cmd_output = Command::new("ps").args(&["-A", "-o", "pid,ppid,args"]).output().unwrap();
    let ps_output = String::from_utf8_lossy(&ps_cmd_output.stdout);
    // On AIX, the PPID is not always present, such as when a process is blocked
    // (marked as <exiting>), or if a process is idle. In these situations,
    // the PPID column contains a "-" for the respective process.
    // Filter out any lines that have a "-" as the PPID as the PPID is
    // expected to be an integer.
    let filtered_ps: Vec<_> =
        ps_output.lines().filter(|line| line.split_whitespace().nth(1) != Some("-")).collect();

    for (line_no, line) in filtered_ps.into_iter().enumerate() {
        if 0 < line_no
            && 0 < line.len()
            && my_pid
                == line
                    .split(' ')
                    .filter(|w| 0 < w.len())
                    .nth(1)
                    .expect("1st column should be PPID")
                    .parse()
                    .ok()
                    .expect("PPID string into integer")
            && line.contains("defunct")
        {
            panic!("Zombie child {}", line);
        }
    }
}

#[cfg(windows)]
fn find_zombies() {}

fn main() {
    let too_long = format!("/NoSuchCommand{:0300}", 0u8);

    let _failures = (0..100)
        .map(|_| {
            let mut cmd = Command::new(&too_long);
            let failed = cmd.spawn();
            assert!(failed.is_err(), "Make sure the command fails to spawn(): {:?}", cmd);
            failed
        })
        .collect::<Vec<_>>();

    find_zombies();
    // then _failures goes out of scope
}
