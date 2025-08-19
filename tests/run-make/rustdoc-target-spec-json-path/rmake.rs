// Test that rustdoc will properly canonicalize the target spec json path just like rustc.
//@ needs-llvm-components: x86

use run_make_support::{cwd, rustc, rustdoc};

fn main() {
    let out_dir = "rustdoc-target-spec-json-path";
    rustc().crate_type("lib").input("dummy_core.rs").target("target.json").run();
    rustdoc()
        .input("my_crate.rs")
        .out_dir(out_dir)
        .library_search_path(cwd())
        .target("target.json")
        .run();
}
