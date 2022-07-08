// run-pass
// needs-unwind
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-wasm32-bare no unwinding panic
// ignore-avr no unwinding panic
// ignore-nvptx64 no unwinding panic

use std::{env, ops::Not, panic, process};

fn main() {
    match &env::args().collect::<Vec<_>>()[..] {
        [just_me] => parent(just_me),
        [_me, subprocess] if subprocess == "panic" => subprocess_panic(),
        [_me, subprocess] if subprocess == "resume_unwind" => subprocess_resume_unwind(),
        _ => unreachable!(),
    }
}

fn parent(self_exe: &str) {
    // call the subprocess 1: panic with a drop bomb
    let status =
        process::Command::new(self_exe)
            .arg("panic")
            .status()
            .expect("running the command should have succeeded")
    ;
    assert!(status.success().not(), "`subprocess_panic()` is expected to have aborted");

    // call the subprocess 2: resume_unwind with a drop bomb
    let status =
        process::Command::new(self_exe)
            .arg("resume_unwind")
            .status()
            .expect("running the command should have succeeded")
    ;
    assert!(status.success().not(), "`subprocess_resume_unwind()` is expected to have aborted");
}

fn subprocess_panic() {
    let _ = panic::catch_unwind(|| {
        struct Bomb;

        impl Drop for Bomb {
            fn drop(&mut self) {
                panic!();
            }
        }

        let panic_payload = panic::catch_unwind(|| panic::panic_any(Bomb)).unwrap_err();
        // Calls `Bomb::drop`, which starts unwinding. But since this is a panic payload already,
        // the drop glue is amended to abort on unwind. So this ought to abort the process.
        drop(panic_payload);
    });
}

fn subprocess_resume_unwind() {
    use panic::resume_unwind;
    let _ = panic::catch_unwind(|| {
        struct Bomb;

        impl Drop for Bomb {
            fn drop(&mut self) {
                panic!();
            }
        }

        let panic_payload = panic::catch_unwind(|| resume_unwind(Box::new(Bomb))).unwrap_err();
        // Calls `Bomb::drop`, which starts unwinding. But since this is a panic payload already,
        // the drop glue is amended to abort on unwind. So this ought to abort the process.
        drop(panic_payload);
    });
}
