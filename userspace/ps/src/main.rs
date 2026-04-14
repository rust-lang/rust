#![no_std]
#![no_main]
extern crate alloc;
use alloc::string::{String, ToString};
use core::prelude::v1::*;

use stem::syscall::{exit, vfs_close, vfs_open, vfs_read, vfs_readdir, vfs_write};

fn print(msg: &str) {
    let _ = vfs_write(1, msg.as_bytes()).ok();
}

fn read_file(path: &str) -> Option<String> {
    let fd = vfs_open(path, 0).ok()?;
    let mut buf = [0u8; 1024];
    let n = vfs_read(fd, &mut buf).ok()?;
    let _ = vfs_close(fd);
    Some(String::from_utf8_lossy(&buf[..n]).into_owned())
}

#[stem::main]
fn main(_arg: usize) -> ! {
    print("  PID  PPID STAT COMMAND\n");

    let fd = match vfs_open("/proc", 0) {
        Ok(fd) => fd,
        Err(_) => {
            print("ps: failed to open /proc\n");
            exit(1);
        }
    };

    let mut buf = [0u8; 4096];
    loop {
        match vfs_readdir(fd, &mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut offset = 0;
                while offset < n {
                    let mut end = offset;
                    while end < n && buf[end] != 0 {
                        end += 1;
                    }
                    if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
                        if !name.is_empty() && name.chars().all(|c| c.is_ascii_digit()) {
                            // It's a PID directory
                            let status_path = alloc::format!("/proc/{}/status", name);
                            if let Some(status) = read_file(&status_path) {
                                let mut pid = "";
                                let mut ppid = "";
                                let mut stat = "";
                                let mut name_val = "";

                                for line in status.lines() {
                                    if line.starts_with("Name:") {
                                        name_val = line.split('\t').nth(1).unwrap_or("").trim();
                                    } else if line.starts_with("State:") {
                                        stat = line.split('\t').nth(1).unwrap_or("").trim();
                                    } else if line.starts_with("Pid:") {
                                        pid = line.split('\t').nth(1).unwrap_or("").trim();
                                    } else if line.starts_with("PPid:") {
                                        ppid = line.split('\t').nth(1).unwrap_or("").trim();
                                    }
                                }

                                // Try to get cmdline for a better command name
                                let cmdline_path = alloc::format!("/proc/{}/cmdline", name);
                                let cmd_display = if let Some(cmdline) = read_file(&cmdline_path) {
                                    if cmdline.is_empty() {
                                        name_val.to_string()
                                    } else {
                                        // cmdline is null-delimited
                                        cmdline.replace('\0', " ").trim().to_string()
                                    }
                                } else {
                                    name_val.to_string()
                                };

                                print(&alloc::format!(
                                    "{:>5} {:>5} {:<4} {}\n",
                                    pid, ppid, stat, cmd_display
                                ));
                            }
                        }
                    }
                    offset = end + 1;
                }
            }
            Err(_) => break,
        }
    }

    let _ = vfs_close(fd).ok();
    exit(0)
}
