//! This test ensures that the call locations are not duplicated when generating scraped examples.
//! To ensure that, we check that this call doesn't fail.
//! Regression test for <https://github.com/rust-lang/rust/issues/153837>.

use run_make_support::{cargo, htmldocck};

fn main() {
    cargo().args(["rustdoc", "-Zunstable-options", "-Zrustdoc-scrape-examples"]).run();

    htmldocck().arg("target/doc").arg("src/lib.rs").run();
}
