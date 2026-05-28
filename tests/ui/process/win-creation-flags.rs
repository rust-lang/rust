// Test that windows `creation_flags` extension to `Command` works.

//@ run-pass
//@ only-windows
//@ needs-subprocess

use std::env;
use std::os::windows::process::CommandExt;
use std::process::{Command, exit};

fn main() {
    if env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let exe = env::current_exe().unwrap();

    // Use the DETACH_PROCESS to create a subprocess that isn't attached to the console.
    // The subprocess's exit status will be 0 if it's detached.
    let status = Command::new(&exe)
        .arg("--child")
        .creation_flags(DETACH_PROCESS)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(status.code(), Some(0));

    // Try without DETACH_PROCESS to ensure this test works.
    let status = Command::new(&exe).arg("--child").spawn().unwrap().wait().unwrap();
    assert_eq!(status.code(), Some(1));
}

// exits with 1 if the console is attached or 0 otherwise
fn child() {
    // Get the attached console's code page.
    // This will fail (return 0) if no console is attached.
    let has_console = GetConsoleCP() != 0;
    exit(has_console as i32);
}

// Windows API definitions.
const DETACH_PROCESS: u32 = 0x00000008;
#[link(name = "kernel32")]
unsafe extern "system" {
    safe fn GetConsoleCP() -> u32;
}
