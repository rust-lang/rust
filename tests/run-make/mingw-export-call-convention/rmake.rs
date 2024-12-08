// On windows-gnu, symbol exporting used to fail to export names
// with no_mangle. #72049 brought this feature up to par with msvc,
// and this test checks that the symbol "bar" is successfully exported.
// See https://github.com/rust-lang/rust/issues/50176

//@ only-x86_64-pc-windows-gnu

use run_make_support::{llvm_readobj, rustc};

fn main() {
    rustc().input("foo.rs").run();
    llvm_readobj().arg("--all").input("libfoo.dll.a").run().assert_stdout_contains("bar");
}
