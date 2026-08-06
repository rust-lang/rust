//@ needs-target-std
//
// Test that the suggestion to constrain a type parameter that is dropped in a const
// function with a `[const] Destruct` bound is only offered on nightly, since the bound
// requires an unstable feature.

use run_make_support::{diff, rustc};

fn main() {
    let out = rustc()
        .input("const-drop.rs")
        .env("RUSTC_BOOTSTRAP", "-1")
        .run_fail()
        .assert_stderr_not_contains("consider restricting type parameter `T`")
        .stderr_utf8();
    diff().expected_file("const-drop-stable.stderr").actual_text("(rustc)", &out).run();
    let out = rustc()
        .input("const-drop.rs")
        .ui_testing()
        .run_fail()
        .assert_stderr_contains(
            "consider restricting type parameter `T` with unstable trait `Destruct`",
        )
        .stderr_utf8();
    diff().expected_file("const-drop-nightly.stderr").actual_text("(rustc)", &out).run();
}
