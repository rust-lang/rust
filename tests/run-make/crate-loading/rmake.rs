//@ only-linux
//@ ignore-wasm32
//@ ignore-wasm64

use run_make_support::rfs::copy;
use run_make_support::{assert_contains, rust_lib_name, rustc};

fn main() {
    rustc().input("multiple-dep-versions-1.rs").run();
    rustc().input("multiple-dep-versions-2.rs").extra_filename("2").metadata("2").run();

    let out = rustc()
        .input("multiple-dep-versions.rs")
        .extern_("dependency", rust_lib_name("dependency"))
        .extern_("dep_2_reexport", rust_lib_name("dependency2"))
        .inspect(|cmd| eprintln!("{cmd:?}"))
        .run_fail();
    let stderr = out.stderr_utf8();
    assert_contains(
        &stderr,
        "you have multiple different versions of crate `dependency` in your dependency graph",
    );
    assert_contains(
        &stderr,
        "two types coming from two different versions of the same crate are different types even \
         if they look the same",
    );
    assert_contains(&stderr, "this type doesn't implement the required trait");
    assert_contains(&stderr, "this type implements the required trait");
    assert_contains(&stderr, "this is the required trait");
}
