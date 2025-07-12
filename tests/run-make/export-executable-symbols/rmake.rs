// The unstable flag `-Z export-executable-symbols` exports symbols from executables, as if
// they were dynamic libraries. This test is a simple smoke test to check that this feature
// works by using it in compilation, then checking that the output binary contains the exported
// symbol.
// See https://github.com/rust-lang/rust/pull/85673

//@ ignore-wasm

use run_make_support::{assert_contains, bin_name, llvm_nm, llvm_readobj, rustc, target};

fn main() {
    let target = target();
    rustc()
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Zexport-executable-symbols")
        .input("main.rs")
        .crate_type("bin")
        .run();
    if target.contains("linux") {
        let output = llvm_nm()
            .arg("--dynamic")
            .arg("--defined-only")
            .input(bin_name("main"))
            .run()
            .invalid_stdout_utf8();
        assert_contains(&output, "exported_symbol");
    } else {
        let output = llvm_readobj().symbols().input(bin_name("main")).run().invalid_stdout_utf8();
        assert_contains(&output, "exported_symbol");
    }
}
