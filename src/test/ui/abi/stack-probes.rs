// run-pass
// ignore-arm
// ignore-aarch64
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// ignore-wasm
// ignore-emscripten no processes
// ignore-sgx no processes

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
            "main-thread" => recurse(&MaybeUninit::uninit()),
            "child-thread" => thread::spawn(|| recurse(&MaybeUninit::uninit())).join().unwrap(),
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
        assert_overflow(Command::new(&me).arg("main-thread"));
    }
    assert_overflow(Command::new(&me).arg("child-thread"));
}

#[allow(unconditional_recursion)]
fn recurse(array: &MaybeUninit<[u64; 1024]>) {
    unsafe {
        black_box(array.as_ptr() as u64);
    }
    let local: MaybeUninit<[u64; 1024]> = MaybeUninit::uninit();
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
