// Test that rustdoc will properly canonicalize the target spec json path just like rustc.

use run_make_support::{rustc, rustdoc, tmp_dir};

fn main() {
    let out_dir = tmp_dir().join("rustdoc-target-spec-json-path");
    rustc().crate_type("lib").input("dummy_core.rs").target("target.json").run();
    rustdoc()
        .input("my_crate.rs")
        .output(out_dir)
        .library_search_path(tmp_dir())
        .target("target.json")
        .run();
}
