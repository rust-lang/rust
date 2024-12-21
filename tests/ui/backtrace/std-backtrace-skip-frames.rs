/* This tests a lot of fiddly platform-specific impl details.
   It will probably flake a lot for a while. Feel free to split it up into different tests or add more `ignore-` directives.
   If at all possible, rather than removing the assert that the different
   revisions behave the same, move only the revisions that are failing into a
   separate test, so that the rest are still kept the same.
 */

//@ ignore-android FIXME #17520
//@ ignore-wasm32 spawning processes is not supported
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-sgx no processes
//@ ignore-i686-pc-windows-msvc see #62897 and `backtrace-debuginfo.rs` test
//@ ignore-fuchsia Backtraces not symbolized

//@ run-pass
//@ check-run-results
//@ normalize-stderr-test: "omitted [0-9]+ frames" -> "omitted N frames"
//@ normalize-stderr-test: ".rs:[0-9]+:[0-9]+" -> ".rs:LL:CC"
//@ error-pattern:stack backtrace:
//@ regex-error-pattern:omitted [0-9]+ frames
//@ error-pattern:main
// NOTE: if this is missing it's probably because the check that .stderr files match failed.
//@ error-pattern:finished all checks

//@ exec-env:RUST_BACKTRACE=1
//@ unset-exec-env:RUST_LIB_BACKTRACE
//@ edition:2021
//@ rustc-env:SOURCE_DIR={{src-base}}
//@ compile-flags:-Cstrip=none -Cdebug-assertions=true --check-cfg=cfg(bootstrap)
//@ revisions:line-tables limited full no-split packed unpacked
//@[no-split] ignore-msvc
//@[unpacked] ignore-msvc

//@[no-split] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=off
//@[packed] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=packed
//@[unpacked] compile-flags:-Cdebuginfo=line-tables-only -Csplit-debuginfo=unpacked
//@[line-tables] compile-flags:-Cdebuginfo=line-tables-only
//@[limited] compile-flags:-Cdebuginfo=limited
//@[full] compile-flags:-Cdebuginfo=full

use std::collections::BTreeMap;
use std::env;
use std::path::Path;

fn main() {
    // Make sure this comes first. Otherwise the error message when the check below fails prevents you from using --bless to see the actual output.
    std::panic::catch_unwind(|| check_all_panics()).unwrap_err();

    // compiletest generates a bunch of files for each revision. make sure they're all the same.
    let mut files = BTreeMap::new();
    let dir = Path::new(env!("SOURCE_DIR")).join("backtrace");
    for file in std::fs::read_dir(dir).unwrap() {
        let file = file.unwrap();
        let name = file.file_name().into_string().unwrap();
        if !file.file_type().unwrap().is_file() || !name.starts_with("std-backtrace-skip-frames.") || !name.ends_with(".run.stderr") {
            continue;
        }
        files.insert(name, std::fs::read_to_string(file.path()).unwrap());
    }

    let mut first_line_tables = None;
    let mut first_full = None;

    for (name, contents) in &files {
        // These have different output. Rather than duplicating this whole test, just special-case them here.
        let target = if name.contains(".full.") || name.contains(".limited.") {
            &mut first_full
        } else {
            &mut first_line_tables
        };
        if let Some((target_name, target_contents)) = target {
            if contents != *target_contents {
                eprintln!("are you *sure* that you want {name} to have different backtrace output than {target_name}?");
                eprintln!("NOTE: this test is stateful; run `rm tests/ui/backtrace/std-backtrace-skip-frames.*.stderr` to reset it");
                std::process::exit(0);
            }
        } else {
            // compiletest doesn't support negative matching for `error-pattern`. Do that here.
            assert!(!contents.contains("FnOnce::call_once"));
            *target = Some((name, contents));
        }
    }

    // We need this so people don't --bless away the assertion failure by accident.
    eprintln!("finished all checks");
}

fn check_all_panics() {
    // Spawn a bunch of threads and make sure all of them hide panic details we don't care about.
    for func in [unwrap_result, expect_result, unwrap_option, expect_option, explicit_panic, literal_panic, /*panic_nounwind*/] {
        std::thread::spawn(move || func()).join().unwrap_err();
    }
    std::thread::spawn(|| panic_fmt(3)).join().unwrap_err();

    // Finally, panic ourselves so we can make sure `lang_start`, etc. frames are hidden.
    panic!();
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
    unsafe { [0].get_unchecked(1); }
}
