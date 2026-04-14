//! Smoke test: current_dir() and set_current_dir() via SYS_FS_GETCWD / SYS_FS_CHDIR.
//!
//! Acceptance criteria:
//!   - stem::syscall::vfs_getcwd() returns a non-empty path
//!   - stem::syscall::vfs_chdir("/tmp") changes the cwd
//!   - a subsequent vfs_getcwd() reflects the change
#![no_std]
#![no_main]
extern crate alloc;
use alloc::string::{String, ToString};
use core::default::Default;

#[stem::main]
fn main() -> ! {
    // Initial cwd
    let mut buf = [0u8; 4096];
    let initial = match stem::syscall::vfs_getcwd(&mut buf) {
        Ok(len) => {
            let p = core::str::from_utf8(&buf[..len]).unwrap_or("");
            stem::println!("[cwd_test] initial cwd: {:?}", p);
            p
        }
        Err(e) => {
            stem::println!("[cwd_test] FAIL: vfs_getcwd() error: {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    };

    if initial.is_empty() {
        stem::println!("[cwd_test] FAIL: initial cwd is empty");
        loop { stem::syscall::exit(1); }
    }

    // Change to /tmp (always exists as a ramfs mount on ThingOS)
    match stem::syscall::vfs_chdir("/tmp") {
        Ok(()) => stem::println!("[cwd_test] chdir /tmp OK"),
        Err(e) => {
            stem::println!("[cwd_test] FAIL: vfs_chdir(/tmp): {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    }

    // Verify cwd changed
    match stem::syscall::vfs_getcwd(&mut buf) {
        Ok(len) => {
            let p = core::str::from_utf8(&buf[..len]).unwrap_or("");
            stem::println!("[cwd_test] new cwd: {:?}", p);
            if p != "/tmp" {
                stem::println!("[cwd_test] FAIL: expected /tmp, got {:?}", p);
                loop { stem::syscall::exit(1); }
            }
        }
        Err(e) => {
            stem::println!("[cwd_test] FAIL: vfs_getcwd() after chdir: {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    }

    stem::println!("[cwd_test] PASS");
    loop { stem::syscall::exit(0); }
}
