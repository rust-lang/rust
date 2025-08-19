//@ ignore-cross-compile
//@ needs-crate-type: proc-macro
//@ ignore-musl (FIXME: can't find `-lunwind`)

// A compiler bug caused the following issue:
// If a crate A depends on crate B, and crate B
// depends on crate C, and crate C contains a procedural
// macro, compiling crate A would fail.
// This was fixed in #37846, and this test checks
// that this bug does not make a resurgence.

use run_make_support::{bare_rustc, cwd, rust_lib_name, rustc};

fn main() {
    rustc().input("a.rs").run();
    rustc().input("b.rs").run();
    let curr_dir = cwd().display().to_string();
    bare_rustc()
        .input("c.rs")
        .arg(format!("-Ldependency={curr_dir}"))
        .extern_("b", cwd().join(rust_lib_name("b")))
        .out_dir(cwd())
        .run();
}
