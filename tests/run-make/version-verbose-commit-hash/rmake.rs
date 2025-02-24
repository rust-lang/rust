//! `--version --verbose` should display the git-commit hashes of rustc and rustdoc, but this
//! functionality was lost due to #104184. After this feature was returned by #109981, this
//! test ensures it will not be broken again.
//! See <https://github.com/rust-lang/rust/issues/107094>.
//!
//! # Important note
//!
//! This test is **not** gated by compiletest, and **cannot** trust bootstrap's git-hash logic e.g.
//! if bootstrap reports git-hash is available yet the built rustc doesn't actually have a hash. It
//! must directly communicate with CI, and gate it being run on an env var expected to be set in CI
//! (or that env var being set locally), `COMPILETEST_HAS_GIT_HASH=1`.

use run_make_support::{bare_rustc, bare_rustdoc, regex};

fn main() {
    if !std::env::var("COMPILETEST_HAS_GIT_HASH").is_ok_and(|v| v == "1") {
        return;
    }

    let out_rustc =
        bare_rustc().arg("--version").arg("--verbose").run().stdout_utf8().to_lowercase();
    let out_rustdoc =
        bare_rustdoc().arg("--version").arg("--verbose").run().stdout_utf8().to_lowercase();
    let re =
        regex::Regex::new(r#"commit-hash: [0-9a-f]{40}\ncommit-date: [0-9]{4}-[0-9]{2}-[0-9]{2}"#)
            .unwrap();

    println!("rustc:\n{}", out_rustc);
    println!("rustdoc:\n{}", out_rustdoc);

    assert!(re.is_match(&out_rustc));
    assert!(re.is_match(&out_rustdoc));
}
