// How to run this
// $ RUSTBUILD_FORCE_CLANG_BASED_TESTS=1 ./x.py test tests/run-make/wasm-override-linker/

//@ needs-force-clang-based-tests

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
