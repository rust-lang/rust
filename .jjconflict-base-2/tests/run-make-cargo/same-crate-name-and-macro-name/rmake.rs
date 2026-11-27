//! Regression test for
//! <https://github.com/rust-lang/rust/issues/71259#issuecomment-615879925>
//! (that particular comment describes the issue well).
//!
//! We test that two library crates with the same name can export macros with
//! the same name without causing interference when both are used in another
//! crate.

use run_make_support::cargo;

fn main() {
    cargo().current_dir("consumer").arg("run").run();
}
