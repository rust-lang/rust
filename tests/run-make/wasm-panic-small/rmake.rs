//@ only-wasm32-wasip1
#![deny(warnings)]

use run_make_support::{rfs, rustc};

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
        .target("wasm32-wasip1")
        .arg("-Clto")
        .arg("-Cstrip=debuginfo")
        .opt()
        .cfg(cfg)
        .run();

    let bytes = rfs::read("foo.wasm");
    println!("{}", bytes.len());
    assert!(bytes.len() < 40_000, "bytes len was: {}", bytes.len());
}
