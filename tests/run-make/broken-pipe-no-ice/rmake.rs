//! Check that `rustc` and `rustdoc` does not ICE upon encountering a broken pipe due to unhandled
//! panics from raw std `println!` usages.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/34376>.

//@ ignore-cross-compile (needs to run test binary)

//@ ignore-apple
// FIXME(#131436): on macOS rustc is still reporting the std broken pipe io error panick but it
// doesn't fail with 101 exit status (it terminates with a wait status of SIGPIPE). It doesn't say
// Internal Compiler Error strangely, but it doesn't even go through normal diagnostic infra. Very
// strange.

use std::io::Read;
use std::process::{Command, Stdio};

use run_make_support::env_var;

#[derive(Debug, PartialEq)]
enum Binary {
    Rustc,
    Rustdoc,
}

fn check_broken_pipe_handled_gracefully(bin: Binary, mut cmd: Command) {
    let (reader, writer) = std::io::pipe().unwrap();
    drop(reader); // close read-end
    cmd.stdout(writer).stderr(Stdio::piped());

    let mut child = cmd.spawn().unwrap();

    let mut stderr = String::new();
    child.stderr.as_mut().unwrap().read_to_string(&mut stderr).unwrap();
    let status = child.wait().unwrap();

    assert!(!status.success(), "{bin:?} unexpectedly succeeded");

    const PANIC_ICE_EXIT_CODE: i32 = 101;

    #[cfg(not(windows))]
    {
        // On non-Windows, rustc/rustdoc built with `-Zon-broken-pipe=kill` shouldn't have an exit
        // code of 101 because it should have an wait status that corresponds to SIGPIPE signal
        // number.
        assert_ne!(status.code(), Some(PANIC_ICE_EXIT_CODE), "{bin:?}");
        // And the stderr should be empty because rustc/rustdoc should've gotten killed.
        assert!(stderr.is_empty(), "{bin:?} stderr:\n{}", stderr);
    }

    #[cfg(windows)]
    {
        match bin {
            // On Windows, rustc has a paper that propagates the panic exit code of 101 but converts
            // broken pipe errors into fatal errors instead of ICEs.
            Binary::Rustc => {
                assert_eq!(status.code(), Some(PANIC_ICE_EXIT_CODE), "{bin:?}");
                // But make sure it doesn't manifest as an ICE.
                assert!(!stderr.contains("internal compiler error"), "{bin:?} ICE'd");
            }
            // On Windows, rustdoc seems to cleanly exit with exit code of 1.
            Binary::Rustdoc => {
                assert_eq!(status.code(), Some(1), "{bin:?}");
                assert!(!stderr.contains("panic"), "{bin:?} stderr contains panic");
            }
        }
    }
}

fn main() {
    let mut rustc = Command::new(env_var("RUSTC"));
    rustc.arg("--print=sysroot");
    check_broken_pipe_handled_gracefully(Binary::Rustc, rustc);

    let mut rustdoc = Command::new(env_var("RUSTDOC"));
    rustdoc.arg("--version");
    check_broken_pipe_handled_gracefully(Binary::Rustdoc, rustdoc);
}
