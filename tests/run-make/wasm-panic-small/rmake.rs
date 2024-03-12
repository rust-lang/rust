#![deny(warnings)]

extern crate run_make_support;

use run_make_support::{out_dir, rustc};

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    test("a");
    test("b");
    test("c");
    test("d");
}

fn test(cfg: &str) {
    eprintln!("running cfg {cfg:?}");
    rustc()
        .arg("foo.rs")
        .arg("--target=wasm32-wasip1")
        .arg("-Clto")
        .arg("-O")
        .arg("--cfg")
        .arg(cfg)
        .run();

    let bytes = std::fs::read(&out_dir().join("foo.wasm")).unwrap();
    println!("{}", bytes.len());
    assert!(bytes.len() < 40_000);
}
