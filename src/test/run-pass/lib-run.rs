// xfail-stage0

use std;
import std::run;

// Regression test for memory leaks
fn test_leaks() {
  run::run_program("echo", []);
  run::start_program("echo", []);
  run::program_output("echo", []);
}

fn main() {
  test_leaks();
}