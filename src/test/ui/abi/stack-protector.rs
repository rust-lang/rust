// run-pass
// only-x86_64-unknown-linux-gnu
// revisions: ssp no-ssp
// [ssp] compile-flags: -Z stack-protector=all
// compile-flags: -C opt-level=2
// compile-flags: -g

use std::env;
use std::process::{Command, ExitStatus};

fn main() {
    if env::args().len() == 1 {
        // The test is initially run without arguments. Start the process again,
        // this time *with* an argument; in this configuration, the test program
        // will deliberately smash the stack.
        let cur_argv0 = env::current_exe().unwrap();
        let mut child = Command::new(&cur_argv0);
        child.arg("stacksmash");

        if cfg!(ssp) {
            assert_stack_smash_prevented(&mut child);
        } else {
            assert_stack_smashed(&mut child);
        }
    } else {
        vulnerable_function();
        // If we return here the test is broken: it should either have called
        // malicious_code() which terminates the process, or be caught by the
        // stack check which also terminates the process.
        panic!("TEST BUG: stack smash unsuccessful");
    }
}

// Avoid inlining to make sure the return address is pushed to stack.
#[inline(never)]
fn vulnerable_function() {
    let mut x = 5usize;
    let stackaddr = &mut x as *mut usize;
    let bad_code_ptr = malicious_code as usize;
    // Overwrite the on-stack return address with the address of `malicious_code()`,
    // thereby jumping to that function when returning from `vulnerable_function()`.
    unsafe { fill(stackaddr, bad_code_ptr, 20); }
}

// Use an uninlined function with its own stack frame to make sure that we don't
// clobber e.g. the counter or address local variable.
#[inline(never)]
unsafe fn fill(addr: *mut usize, val: usize, count: usize) {
    let mut addr = addr;
    for _ in 0..count {
        *addr = val;
        addr = addr.add(1);
    }
}

// We jump to malicious_code() having wreaked havoc with the previous stack
// frame and not setting up a new one. This function is therefore constrained,
// e.g. both println!() and std::process::exit() segfaults if called. We
// therefore keep the amount of work to a minimum by calling POSIX functions
// directly.
// The function is un-inlined just to make it possible to set a breakpoint here.
#[inline(never)]
fn malicious_code() {
    let msg = [112u8, 119u8, 110u8, 101u8, 100u8, 33u8, 0u8]; // "pwned!\0" ascii
    unsafe {
        write(1, &msg as *const u8, msg.len());
        _exit(0);
    }
}
extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn _exit(status: i32) -> !;
}


fn assert_stack_smash_prevented(cmd: &mut Command) {
    let (status, stdout, stderr) = run(cmd);
    assert!(!status.success());
    assert!(stdout.is_empty());
    assert!(stderr.contains("stack smashing detected"));
}

fn assert_stack_smashed(cmd: &mut Command) {
    let (status, stdout, stderr) = run(cmd);
    assert!(status.success());
    assert!(stdout.contains("pwned!"));
    assert!(stderr.is_empty());
}


fn run(cmd: &mut Command) -> (ExitStatus, String, String) {
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("status: {}", output.status);
    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);
    (output.status, stdout.to_string(), stderr.to_string())
}
