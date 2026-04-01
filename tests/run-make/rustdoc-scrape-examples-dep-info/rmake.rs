//@ needs-target-std
use run_make_support::{assert_contains, rfs};

#[path = "../rustdoc-scrape-examples-remap/scrape.rs"]
mod scrape;

fn main() {
    rfs::create_dir("rustdoc");

    scrape::scrape(
        &["--scrape-tests", "--emit=dep-info"],
        &["--emit=dep-info,invocation-specific"],
    );

    let content = rfs::read_to_string("rustdoc/foobar.d").replace(r"\", "/");
    assert_contains(&content, "lib.rs:");
    assert_contains(&content, "rustdoc/ex.calls:");

    let content = rfs::read_to_string("rustdoc/ex.d").replace(r"\", "/");
    assert_contains(&content, "examples/ex.rs:");
}
