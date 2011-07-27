use std;
import std::run;

// Regression test for memory leaks
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[test]
fn test_leaks() {
    run::run_program("echo", []);
    run::start_program("echo", []);
    run::program_output("echo", []);
}

// FIXME
#[cfg(target_os = "win32")]
#[test]
#[ignore]
fn test_leaks() { }