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

//@ run-pass
//@ check-run-results
//@ normalize-stderr: "omitted [0-9]+ frames" -> "omitted N frames"
//@ normalize-stderr: ".rs:[0-9]+:[0-9]+" -> ".rs:LL:CC"
//@ error-pattern:stack backtrace:
//@ regex-error-pattern:omitted [0-9]+ frames
//@ error-pattern:main

//@ exec-env:RUST_BACKTRACE=1
//@ unset-exec-env:RUST_LIB_BACKTRACE
//@ edition:2021
//@ compile-flags:-Cstrip=none -Cdebug-assertions=true
//@ revisions:line-tables limited full no-split packed unpacked
//@[no-split] ignore-msvc
//@[unpacked] ignore-msvc

//@[no-split] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=off
//@[packed] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=packed
//@[unpacked] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=unpacked
//@[line-tables] compile-flags:-Cdebuginfo=line-tables-only
//@[limited] compile-flags:-Cdebuginfo=limited
//@[full] compile-flags:-Cdebuginfo=full

fn main() {
    check_all_panics();
}

fn check_all_panics() {
    // Spawn a bunch of threads and make sure all of them hide panic details we don't care about.
    let tests = [
        unwrap_result,
        expect_result,
        unwrap_option,
        expect_option,
        explicit_panic,
        literal_panic,
        /*panic_nounwind*/
    ];
    for func in tests {
        std::thread::spawn(move || func()).join().unwrap_err();
    }
    std::thread::spawn(|| panic_fmt(3)).join().unwrap_err();

    // Finally, panic ourselves so we can make sure `lang_start`, etc. frames are hidden.
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
