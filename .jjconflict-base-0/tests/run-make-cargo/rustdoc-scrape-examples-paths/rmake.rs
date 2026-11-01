//! Test to ensure that the rustdoc `scrape-examples` feature is not panicking.
//! Regression test for <https://github.com/rust-lang/rust/issues/144752>.

use run_make_support::cargo;
use run_make_support::scoped_run::run_in_tmpdir;

fn main() {
    // We copy the crate to be documented "outside" to prevent documenting
    // the whole compiler.
    std::env::set_current_dir("foo").unwrap();
    run_in_tmpdir(|| {
        // The `scrape-examples` feature is also implemented in `cargo` so instead of reproducing
        // what `cargo` does, better to just let `cargo` do it.
        cargo().args(["doc", "-p", "foo", "-Zrustdoc-scrape-examples"]).run();
    })
}
