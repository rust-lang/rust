//! Test to ensure that the rustdoc `scrape-examples` feature is not panicking.
//! Regression test for <https://github.com/rust-lang/rust/issues/144752>.

use run_make_support::{cargo, path, rfs};

fn main() {
    // We copy the crate to be documented "outside" to prevent documenting
    // the whole compiler.
    let tmp = std::env::temp_dir();
    let test_crate = tmp.join("foo");
    rfs::copy_dir_all(path("foo"), &test_crate);

    // The `scrape-examples` feature is also implemented in `cargo` so instead of reproducing
    // what `cargo` does, better to just let `cargo` do it.
    cargo().current_dir(&test_crate).args(["doc", "-p", "foo", "-Zrustdoc-scrape-examples"]).run();
}
