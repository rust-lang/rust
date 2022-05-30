use std::path::{Path, PathBuf};

use super::*;

fn config() -> Config {
    Config {
        args: vec![],
        target: None,
        stderr_filters: vec![],
        stdout_filters: vec![],
        root_dir: PathBuf::from("$RUSTROOT"),
        mode: Mode::Fail,
        path_filter: vec![],
        program: PathBuf::from("cake"),
        output_conflict_handling: OutputConflictHandling::Error,
    }
}

#[test]
fn issue_2156() {
    let s = r"
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR encountered a dangling reference (address $HEX is unallocated)
}
    ";
    let path = Path::new("$DIR/<dummy>");
    let comments = Comments::parse(&path, s);
    let mut errors = vec![];
    let config = config();
    // Crucially, the intended error string *does* appear in this output, as a quote of the comment itself.
    let stderr = br"
error: Undefined Behavior: type validation failed: encountered a dangling reference (address 0x10 is unallocated)
  --> tests/compile-fail/validity/dangling_ref1.rs:6:29
   |
LL |     let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR encountered a dangling reference (address $HEX is unallocated)
   |                             ^^^^^^^^^^^^^^^^^^^^^^^ type validation failed: encountered a dangling reference (address 0x10 is unallocated)
   |
   = help: this indicates a bug in the program: it performed an invalid operation, and caused Undefined Behavior
   = help: see https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html for further information
           
   = note: inside `main` at tests/compile-fail/validity/dangling_ref1.rs:6:29
note: some details are omitted, run with `MIRIFLAGS=-Zmiri-backtrace=full` for a verbose backtrace
error: aborting due to previous error
    ";
    check_test_result(&path, &config, "", "", &comments, &mut errors, /*stdout*/ br"", stderr);
    // The "OutputDiffers" is because we cannot open the .rs file
    match &errors[..] {
        [Error::OutputDiffers { .. }, Error::PatternNotFound { .. }] => {}
        _ => panic!("not the expected error: {:#?}", errors),
    }
}
