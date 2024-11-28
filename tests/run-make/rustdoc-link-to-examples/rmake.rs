// Test to ensure that intra-doc links work correctly with examples.

use std::path::Path;

use run_make_support::rfs::read_to_string;
use run_make_support::{assert_contains, cargo, path};

fn main() {
    let target_dir = path("target");
    cargo().args(&["doc", "-Zunstable-options", "-Zrustdoc-scrape-examples"]).run();

    let content = read_to_string(target_dir.join("doc/foo/index.html"));
    assert_contains(
        &content,
        r#"<a href="../src/check/check.rs.html" title="Example check">check</a>"#,
    );
    assert_contains(
        &content,
        r#"<a href="../src/check/check.rs.html" title="Example check">check/check.rs</a>"#,
    );
    assert_contains(
        &content,
        r#"<a href="../src/check/sub.rs.html" title="Example check">check/sub.rs</a>"#,
    );
}
