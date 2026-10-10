//@ only-wasm32
#![deny(warnings)]

use run_make_support::{rfs, rustc, wasm};

fn main() {
    rustc().input("foo.rs").arg("-Clto").emit_wasm_core_module().opt().run();

    let bytes = rfs::read("foo.wasm");
    let size = wasm::wasm_binary_size(&bytes);
    println!("{size}");
    assert!(size < 75_000);
}
