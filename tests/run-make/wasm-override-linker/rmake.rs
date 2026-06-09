// How to run this
// $ RUSTBUILD_FORCE_CLANG_BASED_TESTS=1 ./x.py test tests/run-make/wasm-override-linker/

//@ needs-force-clang-based-tests
// FIXME(#126180): This test can only run on `x86_64-gnu-debug`, because that CI job sets
// RUSTBUILD_FORCE_CLANG_BASED_TESTS and only runs tests which contain "clang" in their
// name.
// However, this test does not run at all as its name does not contain "clang".

use run_make_support::{env_var, rustc, target};

fn main() {
    if matches!(target().as_str(), "wasm32-unknown-unknown" | "wasm64-unknown-unknown") {
        rustc()
            .input("foo.rs")
            .crate_type("cdylib")
            .target(&target())
            .linker(&env_var("CLANG"))
            .run();
    }
}
