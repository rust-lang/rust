#![deny(warnings)]

extern crate run_make_support;

use run_make_support::{rustc, tmp_dir};

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().input("foo.rs").target("wasm32-wasip1").arg("-Clto").opt().run();

    let bytes = std::fs::read(&tmp_dir().join("foo.wasm")).unwrap();
    println!("{}", bytes.len());
    assert!(bytes.len() < 50_000);
}
