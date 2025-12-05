//@ needs-target-std
use run_make_support::{assert_contains, rfs};

#[path = "../rustdoc-scrape-examples-remap/scrape.rs"]
mod scrape;

fn main() {
    scrape::scrape(
        &["--scrape-tests", "--emit=dep-info"],
        &["--emit=dep-info,invocation-specific"],
    );

    let content = rfs::read_to_string("foobar.d").replace(r"\", "/");
    assert_contains(&content, "lib.rs:");
    assert_contains(&content, "rustdoc/ex.calls:");

    let content = rfs::read_to_string("ex.d").replace(r"\", "/");
    assert_contains(&content, "examples/ex.rs:");
}
