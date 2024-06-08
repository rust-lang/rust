//@ only-wasm32-wasip1
#![deny(warnings)]

use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").arg("-Clto").opt().run();

    let bytes = std::fs::read("foo.wasm").unwrap();
    println!("{}", bytes.len());
    assert!(bytes.len() < 50_000);
}
