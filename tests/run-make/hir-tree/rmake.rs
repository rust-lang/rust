extern crate run_make_support;

use run_make_support::{out_dir, rustc};

// Test that hir-tree output doesn't crash and includes
// the string constant we would expect to see.

fn main() {
    rustc()
        .arg("-o")
        .arg(&out_dir().join("input.hir"))
        .arg("-Zunpretty=hir-tree")
        .arg("input.rs")
        .run();
    let file = std::fs::read_to_string(&out_dir().join("input.hir")).unwrap();
    assert!(file.contains(r#""Hello, Rustaceans!\n""#));
}
