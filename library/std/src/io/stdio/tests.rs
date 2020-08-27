use super::*;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::thread;

#[test]
fn stdout_unwind_safe() {
    assert_unwind_safe::<Stdout>();
}
#[test]
fn stdoutlock_unwind_safe() {
    assert_unwind_safe::<StdoutLock<'_>>();
    assert_unwind_safe::<StdoutLock<'static>>();
}
#[test]
fn stderr_unwind_safe() {
    assert_unwind_safe::<Stderr>();
}
#[test]
fn stderrlock_unwind_safe() {
    assert_unwind_safe::<StderrLock<'_>>();
    assert_unwind_safe::<StderrLock<'static>>();
}

fn assert_unwind_safe<T: UnwindSafe + RefUnwindSafe>() {}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn panic_doesnt_poison() {
    thread::spawn(|| {
        let _a = stdin();
        let _a = _a.lock();
        let _a = stdout();
        let _a = _a.lock();
        let _a = stderr();
        let _a = _a.lock();
        panic!();
    })
    .join()
    .unwrap_err();

    let _a = stdin();
    let _a = _a.lock();
    let _a = stdout();
    let _a = _a.lock();
    let _a = stderr();
    let _a = _a.lock();
}
