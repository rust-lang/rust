//! On Windows the GetExitCodeProcess API is used to get the exit code of a
//! process, but it's easy to mistake a process exiting with the code 259 as
//! "still running" because this is the value of the STILL_ACTIVE constant. Make
//! sure we handle this case in the standard library and correctly report the
//! status.
//!
//! Note that this is disabled on unix as processes exiting with 259 will have
//! their exit status truncated to 3 (only the lower 8 bits are used).

//@ run-pass

#[cfg(windows)]
fn main() {
    use std::env;
    use std::process::{self, Command};

    if env::args().len() == 1 {
        let status = Command::new(env::current_exe().unwrap()).arg("foo").status().unwrap();
        assert_eq!(status.code(), Some(259));
    } else {
        process::exit(259);
    }
}

#[cfg(not(windows))]
fn main() {}
