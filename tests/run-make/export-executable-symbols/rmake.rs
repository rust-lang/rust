// The unstable flag `-Z export-executable-symbols` exports symbols from executables, as if
// they were dynamic libraries. This test is a simple smoke test to check that this feature
// works by using it in compilation, then checking that the output binary contains the exported
// symbol.
// See https://github.com/rust-lang/rust/pull/85673

//@ only-unix
// Reason: the export-executable-symbols flag only works on Unix
// due to hardcoded platform-specific implementation
// (See #85673)
//@ ignore-wasm32
//@ ignore-wasm64
//@ needs-target-std

use run_make_support::{bin_name, llvm_readobj, rustc};

fn main() {
    rustc().arg("-Zexport-executable-symbols").input("main.rs").crate_type("bin").run();
    llvm_readobj()
        .symbols()
        .input(bin_name("main"))
        .run()
        .assert_stdout_contains("exported_symbol");
}
