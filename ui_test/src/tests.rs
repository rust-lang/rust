use std::path::{Path, PathBuf};

use super::{check_annotations, Comments, Config, Error, Mode, OutputConflictHandling};

fn config() -> Config {
    Config {
        args: vec![],
        target: None,
        stderr_filters: vec![],
        stdout_filters: vec![],
        root_dir: PathBuf::from("."),
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
    let comments = Comments::parse(Path::new("<dummy>"), s);
    let mut errors = vec![];
    let config = config();
    let unnormalized_stderr = r"
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
    check_annotations(unnormalized_stderr, &mut errors, &config, "", &comments);
    match &errors[..] {
        [Error::PatternNotFound { .. }] => {}
        _ => panic!("not the expected error: {:#?}", errors),
    }
}
