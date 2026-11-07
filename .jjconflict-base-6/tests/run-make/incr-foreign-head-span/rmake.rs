// Ensure that modifying a crate on disk (without recompiling it)
// does not cause ICEs (internal compiler errors) in downstream crates.
// Previously, we would call `SourceMap.guess_head_span` on a span
// from an external crate, which would cause us to read an upstream
// source file from disk during compilation of a downstream crate.
// See https://github.com/rust-lang/rust/issues/86480

//@ needs-target-std

use run_make_support::{rfs, rust_lib_name, rustc};

fn main() {
    rustc().input("first_crate.rs").incremental("incr").crate_type("lib").run();
    rustc()
        .input("second_crate.rs")
        .incremental("incr")
        .extern_("first_crate", rust_lib_name("first_crate"))
        .crate_type("lib")
        .run();
    rfs::remove_file("first_crate.rs");
    rustc().input("second_crate.rs").incremental("incr").cfg("second_run").crate_type("lib").run();
}
