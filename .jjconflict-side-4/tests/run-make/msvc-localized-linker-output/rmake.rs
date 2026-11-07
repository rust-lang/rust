//@ only-msvc
//@ ignore-cross-compile (need to run the fake link.exe on the host)

//! Tests that localized (non-English) MSVC `link.exe` progress messages are
//! classified as `linker_info`, not `linker_messages`.
//!
//! `link.exe` is hardcoded by rustc to run with `VSLANG=1033`, which only works
//! when an English language pack is installed. Without it, messages like
//! "Creating library ..." are printed in another language, and the English
//! string matching that used to detect them fails. Since all real diagnostics
//! carry a locale-independent `LNK####` code, printed in the structured
//! `LINK : warning LNK####:` form, any line without one is informational, no
//! matter the language it was printed in.

use run_make_support::{bare_rustc, rustc, target};

fn main() {
    // rustc prepends the sysroot's tools bin directory to the linker's `PATH`,
    // which bare names like `link.exe` are resolved against. Put the fake
    // `link.exe` there so it wins over the real linker; `-L` below keeps std
    // available from the real sysroot.
    let fake_sysroot = std::env::current_dir().unwrap().join("fake-sysroot");
    let tools_bin = fake_sysroot.join(format!("lib/rustlib/{}/bin", target()));
    std::fs::create_dir_all(&tools_bin).unwrap();
    rustc().arg("fake-linker.rs").output(tools_bin.join("link.exe")).run();

    let real_libdir = rustc().print("target-libdir").run().stdout_utf8();
    let real_libdir = real_libdir.trim();

    let fake_link = |extra: &[&str]| {
        let mut r = bare_rustc();
        r.input("main.rs")
            .output("main")
            .arg(format!("--sysroot={}", fake_sysroot.display()))
            .arg(format!("-L{real_libdir}"))
            // Matched by name against the linker's `PATH`, so the fake in the
            // tools bin directory is used instead of the real VS linker.
            .arg("-Clinker=link.exe")
            // Overrides `rust.lld=true` on CI.
            .arg("-Clinker-flavor=msvc");
        for a in extra {
            r.arg(a);
        }
        r
    };

    // The localized progress line must not warn by default.
    fake_link(&[])
        .run()
        .assert_stderr_not_contains("linker stdout")
        .assert_stderr_not_contains("ライブラリ foo.dll.lib とオブジェクト foo.dll.exp を作成中");

    // It is still visible through `linker_info`, and must not be misclassified
    // as `linker_messages`.
    fake_link(&["-Wlinker_info", "-Dlinker_messages"]) // Fail if the message is misclassified.
        .run()
        .assert_stderr_contains("ライブラリ foo.dll.lib とオブジェクト foo.dll.exp を作成中");

    // Real diagnostics keep their `LNK####` code and still warn.
    fake_link(&["-Clink-arg=run_make_lnk"])
        .run()
        .assert_stderr_contains(
            "warning: linker stdout: LINK : warning LNK2001: unresolved external symbol foo",
        )
        // The informational LNK6004 line stays hidden.
        .assert_stderr_not_contains("LNK6004");
}
