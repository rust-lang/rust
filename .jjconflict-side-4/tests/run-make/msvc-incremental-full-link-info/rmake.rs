//@ only-msvc
//! Tests that the MSVC incremental linker's "performing full link" message is classified as
//! `linker_info`, not `linker_messages`.
//!
//! The MSVC incremental linker prints this message to stdout when it finds a `.ilk` file for
//! incremental linking  but its associated `.exe` file is missing:
//!
//! ```
//! LINK : ...\main.exe not found or not built by the last incremental link; performing full link
//! ```

use std::fs;

use run_make_support::bare_rustc;

fn incremental_rustc() -> run_make_support::Rustc {
    let mut r = bare_rustc();
    r.input("main.rs")
        // Overrides `rust.lld=true` on CI.
        .arg("-Clinker=link.exe")
        .arg("-Clinker-flavor=msvc")
        // Without this, Rust passes /OPT:REF to link.exe, which
        // disables incremental linking entirely and suppresses the message.
        .arg("-Clink-dead-code")
        // /DEBUG is required: it implies /INCREMENTAL, which is what makes
        // link.exe create the .ilk file and later emit the message.
        .arg("-Cdebuginfo=2");
    r
}

fn main() {
    // First link: produces main.exe and main.ilk.
    incremental_rustc().run();

    // Delete the .exe but leave the .ilk.
    fs::remove_file("main.exe").unwrap();

    // Second link: link.exe finds the .ilk but not the .exe and emits the
    // "performing full link" message on stdout.
    let output = incremental_rustc()
        .arg("-Dlinker_messages") // Fail if the message is misclassified.
        .arg("-Wlinker_info")
        .run();

    // Verify the message was actually emitted and routed to linker_info.
    output.assert_stderr_contains("performing full link");
}
