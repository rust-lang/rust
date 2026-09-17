//@ only-wasm32
#![deny(warnings)]

use run_make_support::{rfs, rustc, wasm};

fn main() {
    test("a");
    test("b");
    test("c");
    test("d");
}

fn test(cfg: &str) {
    eprintln!("running cfg {cfg:?}");

    rustc()
        .input("foo.rs")
        .arg("-Clto")
        .arg("-Cpanic=abort")
        .arg("-Cstrip=debuginfo")
        .opt()
        .cfg(cfg)
        .run();

    let bytes = rfs::read("foo.wasm");
    let size = wasm::wasm_binary_size(&bytes);
    println!("{size}");
    assert!(size < 60_000, "bytes len was: {size}");
}
