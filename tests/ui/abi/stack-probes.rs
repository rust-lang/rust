// run-pass
// ignore-arm
// ignore-aarch64
// ignore-mips
// ignore-mips64
// ignore-sparc
// ignore-sparc64
// ignore-wasm
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-fuchsia no exception handler registered for segfault
// ignore-nto Crash analysis impossible at SIGSEGV in QNX Neutrino

use std::env;
use std::mem::MaybeUninit;
use std::process::Command;
use std::thread;

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    #[link_name = "rust_dbg_extern_identity_u64"]
    fn black_box(u: u64);
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 0 {
        match &args[0][..] {
            "main-recurse" => overflow_recurse(),
            "child-recurse" => thread::spawn(overflow_recurse).join().unwrap(),
            "child-frame" => overflow_frame(),
            _ => panic!(),
        }
        return;
    }

    let me = env::current_exe().unwrap();

    // The linux kernel has some different behavior for the main thread because
    // the main thread's stack can typically grow. We can't always guarantee
    // that we report stack overflow on the main thread, see #43052 for some
    // details
    if cfg!(not(target_os = "linux")) {
        assert_overflow(Command::new(&me).arg("main-recurse"));
    }
    assert_overflow(Command::new(&me).arg("child-recurse"));
    assert_overflow(Command::new(&me).arg("child-frame"));
}

#[allow(unconditional_recursion)]
fn recurse(array: &MaybeUninit<[u64; 1024]>) {
    unsafe {
        black_box(array.as_ptr() as u64);
    }
    let local: MaybeUninit<[u64; 1024]> = MaybeUninit::uninit();
    recurse(&local);
}

#[inline(never)]
fn overflow_recurse() {
    recurse(&MaybeUninit::uninit());
}

fn overflow_frame() {
    // By using a 1MiB stack frame with only 512KiB stack, we'll jump over any
    // guard page, even with 64K pages -- but stack probes should catch it.
    const STACK_SIZE: usize = 512 * 1024;
    thread::Builder::new().stack_size(STACK_SIZE).spawn(|| {
        let local: MaybeUninit<[u8; 2 * STACK_SIZE]> = MaybeUninit::uninit();
        unsafe {
            black_box(local.as_ptr() as u64);
        }
    }).unwrap().join().unwrap();
}

fn assert_overflow(cmd: &mut Command) {
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("status: {}", output.status);
    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);
    assert!(stdout.is_empty());
    assert!(stderr.contains("has overflowed its stack\n"));
}
