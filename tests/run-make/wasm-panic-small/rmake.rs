//@ only-wasm32-wasip1
#![deny(warnings)]

use run_make_support::{fs_wrapper, rustc};

fn main() {
    test("a");
    test("b");
    test("c");
    test("d");
}

fn test(cfg: &str) {
    eprintln!("running cfg {cfg:?}");

    rustc().input("foo.rs").target("wasm32-wasip1").arg("-Clto").opt().cfg(cfg).run();

    let bytes = fs_wrapper::read("foo.wasm");
    println!("{}", bytes.len());
    assert!(bytes.len() < 40_000);
}
