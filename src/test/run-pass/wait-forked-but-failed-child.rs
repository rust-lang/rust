// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

#![feature(rustc_private)]

extern crate libc;

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
    let my_pid = unsafe { libc::getpid() };

    // http://pubs.opengroup.org/onlinepubs/9699919799/utilities/ps.html
    let ps_cmd_output = Command::new("ps").args(&["-A", "-o", "pid,ppid,args"]).output().unwrap();
    let ps_output = String::from_utf8_lossy(&ps_cmd_output.stdout);

    for (line_no, line) in ps_output.split('\n').enumerate() {
        if 0 < line_no && 0 < line.len() &&
           my_pid == line.split(' ').filter(|w| 0 < w.len()).nth(1)
                         .expect("1st column should be PPID")
                         .parse().ok()
                         .expect("PPID string into integer") &&
           line.contains("defunct") {
            panic!("Zombie child {}", line);
        }
    }
}

#[cfg(windows)]
fn find_zombies() { }

fn main() {
    let too_long = format!("/NoSuchCommand{:0300}", 0u8);

    let _failures = (0..100).map(|_| {
        let mut cmd = Command::new(&too_long);
        let failed = cmd.spawn();
        assert!(failed.is_err(), "Make sure the command fails to spawn(): {:?}", cmd);
        failed
    }).collect::<Vec<_>>();

    find_zombies();
    // then _failures goes out of scope
}
