// xfail-stage0

use std;
import std::run;

// Regression test for memory leaks
// FIXME (714) Why does this fail on win32?

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
fn test_leaks() {
  run::run_program("echo", []);
  run::start_program("echo", []);
  run::program_output("echo", []);
}

#[cfg(target_os = "win32")]
fn test_leaks() {}

fn main() {
  test_leaks();
}