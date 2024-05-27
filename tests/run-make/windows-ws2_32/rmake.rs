//@ only-msvc

// Tests that WS2_32.dll is not unnecessarily linked, see issue #85441

use run_make_support::{llvm_readobj, rustc, tmp_dir};

fn main() {
    rustc().input("empty.rs").run();
    let empty = tmp_dir().join("empty.exe");
    let output = llvm_readobj().input(empty).coff_imports().run();
    let output = String::from_utf8(output.stdout).unwrap();
    assert!(!output.to_ascii_uppercase().contains("WS2_32.DLL"));
}
