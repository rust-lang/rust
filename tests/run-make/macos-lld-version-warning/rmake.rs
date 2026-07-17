//! Tests that lld-format version mismatch warnings on macOS are routed to `linker_info`
//! (not `linker_messages`), matching the treatment of ld64/ld_prime deployment warnings.
//! See <https://github.com/rust-lang/rust/issues/159227>

//@ only-macos
//@ ignore-cross-compile (need to run fake linker)

use run_make_support::{diff, rustc};

fn main() {
    rustc().arg("fake-linker.rs").output("fake-linker").run();

    // Use darwin-lld-cc flavor to exercise the Darwin(_, Lld::Yes) path.
    let warnings = rustc()
        .input("main.rs")
        .arg("-Clink-self-contained=-linker")
        .arg("-Zunstable-options")
        .arg("-Clinker-flavor=darwin-lld-cc")
        .linker("./fake-linker")
        .run()
        .stderr_utf8();

    // The lld version warning should appear under linker_info, not linker_messages.
    diff()
        .expected_file("warnings.txt")
        .actual_text("(lld version warning)", &warnings)
        .run();
}
