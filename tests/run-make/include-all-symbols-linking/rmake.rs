// Linkers treat archives differently from object files: all object files participate in linking,
// while archives will only participate in linking if they can satisfy at least one undefined
// reference (version scripts doesn't count). This causes `#[no_mangle]` or `#[used]` items to
// be ignored by the linker, and since they never participate in the linking, using `KEEP` in the
// linker scripts can't keep them either. This causes #47384. After the fix in #95604, this test
// checks that these symbols and sections successfully appear in the output dynamic library.
// See https://github.com/rust-lang/rust/pull/95604
// See https://github.com/rust-lang/rust/issues/47384

//@ needs-target-std
//@ ignore-wasm differences in object file formats causes errors in the llvm_objdump step.
//@ ignore-windows differences in object file formats causes errors in the llvm_objdump step.

use run_make_support::{dynamic_lib_name, llvm_objdump, llvm_readobj, rustc, target};

fn main() {
    rustc().crate_type("lib").input("lib.rs").run();
    let mut main = rustc();
    main.crate_type("cdylib");
    if target().contains("linux") {
        main.link_args("-Tlinker.ld");
    }
    main.input("main.rs").run();

    // Ensure `#[used]` and `KEEP`-ed section is there
    llvm_objdump()
        .arg("--full-contents")
        .arg("--section=.static")
        .input(dynamic_lib_name("main"))
        .run();
    // Ensure `#[no_mangle]` symbol is there
    llvm_readobj()
        .arg("--symbols")
        .input(dynamic_lib_name("main"))
        .run()
        .assert_stdout_contains("bar");
}
