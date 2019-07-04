// ignore-arm
// ignore-aarch64
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// ignore-wasm
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-musl FIXME #31506

use std::mem;
use std::process::Command;
use std::thread;
use std::env;

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    #[link_name = "rust_dbg_extern_identity_u64"]
    fn black_box(u: u64);
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 0 {
        match &args[0][..] {
            "main-thread" => recurse(&[]),
            "child-thread" => thread::spawn(|| recurse(&[])).join().unwrap(),
            _ => panic!(),
        }
        return
    }

    let me = env::current_exe().unwrap();

    // The linux kernel has some different behavior for the main thread because
    // the main thread's stack can typically grow. We can't always guarantee
    // that we report stack overflow on the main thread, see #43052 for some
    // details
    if cfg!(not(target_os = "linux")) {
        assert_overflow(Command::new(&me).arg("main-thread"));
    }
    assert_overflow(Command::new(&me).arg("child-thread"));
}

#[allow(unconditional_recursion)]
fn recurse(array: &[u64]) {
    unsafe { black_box(array.as_ptr() as u64); }
    #[allow(deprecated)]
    let local: [_; 1024] = unsafe { mem::uninitialized() };
    recurse(&local);
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
