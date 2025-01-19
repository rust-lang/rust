/* This tests a lot of fiddly platform-specific impl details.
  It will probably flake a lot for a while. Feel free to split it up into
  different tests or add more `ignore-` directives.
*/

//@ ignore-android FIXME #17520
//@ ignore-wasm32 spawning processes is not supported
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-sgx no processes
//@ ignore-i686-pc-windows-msvc see #62897 and `backtrace-debuginfo.rs` test
//@ ignore-fuchsia Backtraces not symbolized
//@ needs-unwind

//@ check-run-results
//@ normalize-stderr: "omitted [0-9]+ frames?" -> "omitted N frames"
//@ normalize-stderr: ".rs:[0-9]+:[0-9]+" -> ".rs:LL:CC"
//@ error-pattern:stack backtrace:
//@ regex-error-pattern:omitted [0-9]+ frames
// FIXME: what i actually want to do is check that we never have a `0: main` frame.
// this doesn't do that, it just checks that we have *at least* one main frame that isn't 0.
//@ regex-error-pattern:[1-9][0-9]*: .*main

//@ exec-env:RUST_BACKTRACE=1
//@ unset-exec-env:RUST_LIB_BACKTRACE
//@ edition:2021
//@ compile-flags:-Cstrip=none -Cdebug-assertions=true
//@ revisions:line-tables limited full no-split packed unpacked
//@[no-split] ignore-msvc ignore-macos
//@[unpacked] ignore-msvc

//@[no-split] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=off
//@[packed] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=packed
//@[unpacked] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=unpacked
//@[line-tables] compile-flags:-Cdebuginfo=line-tables-only
//@[limited] compile-flags:-Cdebuginfo=limited
//@[full] compile-flags:-Cdebuginfo=full

use std::env;
use std::io::Write;
use std::process::Command;

const UNWRAP_RESULT: &str = "unwrap_result";
const EXPECT_RESULT: &str = "expect_result";
const UNWRAP_OPTION: &str = "unwrap_option";
const EXPECT_OPTION: &str = "expect_option";
const LITERAL_PANIC: &str = "literal_panic";
const EXPLICIT_PANIC: &str = "explicit_panic";
const PANIC_FMT: &str = "panic_fmt";
const PANIC_NOUNWIND: &str = "panic_nounwind";

fn main() {
    let func = match env::args().skip(1).next().as_deref() {
        None => {
            check_all_panics();
        }
        Some(UNWRAP_RESULT) => unwrap_result,
        Some(EXPECT_RESULT) => expect_result,
        Some(UNWRAP_OPTION) => unwrap_option,
        Some(EXPECT_OPTION) => expect_option,
        Some(LITERAL_PANIC) => literal_panic,
        Some(EXPLICIT_PANIC) => explicit_panic,
        Some(PANIC_FMT) => || panic_fmt(3),
        Some(PANIC_NOUNWIND) => panic_nounwind,
        Some(func) => unreachable!("unknown stack trace to check: {func}"),
    };
    // work around https://github.com/rust-lang/rust/issues/134909
    // FIXME: this is really not ideal, the whole point is to test optimized backtraces.
    std::hint::black_box(func)();
}

fn check_all_panics() {
    // Spawn a bunch of processes and make sure all of them hide panic details we don't care about.
    // NOTE: we can't just spawn a thread because cranelift doesn't support unwinding.
    let tests = [
        UNWRAP_RESULT,
        EXPECT_RESULT,
        UNWRAP_OPTION,
        EXPECT_OPTION,
        EXPLICIT_PANIC,
        LITERAL_PANIC,
        PANIC_FMT,
        PANIC_NOUNWIND,
    ];
    for func in tests {
        let me = env::current_exe().unwrap();
        // dbg!(&me);
        let mut cmd = Command::new(me);
        cmd.arg(func);
        let output = cmd.output().unwrap();
        assert!(!output.status.success());
        assert_ne!(output.stderr.len(), 0);
        assert_eq!(output.stdout.len(), 0);
        eprintln!("{func}:");
        std::io::stderr().write_all(&output.stderr).unwrap();
        eprintln!();
        // std::thread::spawn(move || func()).join().unwrap_err();
    }
    // std::thread::spawn(|| panic_fmt(3)).join().unwrap_err();

    // Finally, panic ourselves so we can make sure `lang_start`, etc. frames are hidden.
    // We use catch_unwind just so we can see what the backtrace looks like;
    // cranelift doesn't support unwinding but that's ok because this is the
    // last thing in the file.
    // FIXME: for some reason, adding `unwrap_err` here completely messes up backtraces (???),
    // llvm doesn't generate for inlined functions at all.
    // just ignore the error instead.
    std::panic::catch_unwind(|| panic!()).unwrap_err();
}

fn unwrap_result() {
    Err(()).unwrap()
}

fn expect_result() {
    Err(()).expect("oops")
}

fn unwrap_option() {
    Option::None.unwrap()
}

fn expect_option() {
    Option::None.expect("oops")
}

fn explicit_panic() {
    panic!()
}

fn literal_panic() {
    panic!("oopsie")
}

fn panic_fmt(x: u32) {
    panic!("{x}")
}

#[allow(unused_must_use)]
// TODO: separate test
#[allow(dead_code)]
fn panic_nounwind() {
    unsafe {
        [0].get_unchecked(1);
    }
}
