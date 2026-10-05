//@ needs-target-std
//@ ignore-backends: gcc

// Test that the suggestion to constrain a type parameter that is dropped in a const
// function with a `[const] Destruct` bound is only offered on nightly, since the bound
// requires an unstable feature.

use run_make_support::{diff, rustc, stable_bare_rustc};

fn main() {
    let out = stable_bare_rustc()
        .input("const-drop.rs")
        .edition("2015")
        .run_fail()
        .assert_stderr_not_contains("consider restricting type parameter `T`")
        .stderr_utf8();
    diff().expected_file("const-drop-stable.stderr").actual_text("(rustc)", &out).run();
    let out = rustc()
        .input("const-drop.rs")
        .edition("2015")
        .ui_testing()
        .run_fail()
        .assert_stderr_contains(
            "consider restricting type parameter `T` with unstable trait `Destruct`",
        )
        .stderr_utf8();
    diff().expected_file("const-drop-nightly.stderr").actual_text("(rustc)", &out).run();
}
