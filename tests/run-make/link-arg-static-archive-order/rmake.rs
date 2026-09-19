// Tests for the `linker_static_archive_order` lint.
// See <https://github.com/rust-lang/rust/issues/154975>.

//@ only-linux
// The lint applies to GNU `ld`-family linkers (including `lld`) on non-Windows, non-Darwin.

use run_make_support::{cwd, is_darwin, is_windows, rustc};

fn main() {
    // The diagnostic carries the full archive path, so assert on the stable filename substring.
    let archive = cwd().join("libdoesnotexist.a");
    let archive_str = archive.to_str().unwrap();

    // 1. A `.a` path via `-Clink-arg` on a bare `ld` flavor fires the lint.
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg(archive_str)
        .arg("-Wlinker-static-archive-order")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_contains("warning: static archive")
        .assert_stderr_contains("libdoesnotexist.a");

    // 2. A `.o` path is flagged the same way.
    let object = cwd().join("doesnotexist.o");
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg(object.to_str().unwrap())
        .arg("-Wlinker-static-archive-order")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_contains("warning: static archive")
        .assert_stderr_contains("doesnotexist.o");

    // 3. A non-archive arg like `-lm` does not fire.
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg("-lm")
        .arg("-Wlinker-static-archive-order")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_not_contains("linker_static_archive_order");

    // 4. `Allow` by default: no `-W`, no output.
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg(archive_str)
        .print("link-args")
        .run_unchecked()
        .assert_stderr_not_contains("linker_static_archive_order");

    // 5. `ignore_deny_warnings`: `-Dwarnings` does not promote the `-W` to an error.
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg(archive_str)
        .arg("-Wlinker-static-archive-order")
        .arg("-Dwarnings")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_contains("warning: static archive")
        .assert_stderr_contains("the `linker_static_archive_order` lint ignores `-D warnings`");

    // 6. `-Dlinker-static-archive-order` (specific) does promote to an error.
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg(archive_str)
        .arg("-Dlinker-static-archive-order")
        .print("link-args")
        .run_fail()
        .assert_stderr_contains("error: static archive")
        .assert_stderr_contains("libdoesnotexist.a");

    // 7. `#![allow]` suppresses even an explicit `-W`.
    rustc()
        .input("-")
        .linker_flavor("ld")
        .link_arg(archive_str)
        .arg("-Wlinker-static-archive-order")
        .arg("--crate-type=lib")
        .stdin_buf("#![allow(linker_static_archive_order)] fn main() {}")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_not_contains("static archive");

    // 8. The `gnu-cc` flavor (cc-driven, `Gnu(Cc::Yes, Lld::No)`) also fires: it uses real `ld`
    //    behind `cc` and the same `--as-needed` + left-to-right interaction applies.
    rustc()
        .input("empty.rs")
        .arg("-Zunstable-options")
        .linker_flavor("gnu-cc")
        .link_arg(archive_str)
        .arg("-Wlinker-static-archive-order")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_contains("warning: static archive")
        .assert_stderr_contains("libdoesnotexist.a");

    // 9. The lint also fires with `lld` (`Gnu(.., Lld::Yes)`): the risky ordering is the same, so
    //    the latent issue surfaces before a switch back to `ld.bfd`.
    rustc()
        .input("empty.rs")
        .linker_flavor("ld.lld")
        .link_arg(archive_str)
        .arg("-Wlinker-static-archive-order")
        .print("link-args")
        .run_unchecked()
        .assert_stderr_contains("warning: static archive")
        .assert_stderr_contains("libdoesnotexist.a");

    assert!(!is_windows() && !is_darwin());
}
