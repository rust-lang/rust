// The unstable flag `-Z export-executable-symbols` exports symbols from executables, as if
// they were dynamic libraries. This test is a simple smoke test to check that this feature
// works by using it in compilation, then checking that the output binary contains the exported
// symbol.
// See https://github.com/rust-lang/rust/pull/85673

//@ ignore-wasm
//@ ignore-cross-compile

use run_make_support::object::Object;
use run_make_support::{bin_name, is_darwin, object, rustc};

fn main() {
    rustc()
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Zexport-executable-symbols")
        .input("main.rs")
        .crate_type("bin")
        .run();
    let name: &[u8] = if is_darwin() { b"_exported_symbol" } else { b"exported_symbol" };
    let contents = std::fs::read(bin_name("main")).unwrap();
    let object = object::File::parse(contents.as_slice()).unwrap();
    let found = object.exports().unwrap().iter().any(|x| x.name() == name);
    assert!(found);
}
