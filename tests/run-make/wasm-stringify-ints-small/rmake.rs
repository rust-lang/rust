//@ only-wasm32-wasip1
#![deny(warnings)]

use run_make_support::{fs_wrapper, rustc};

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").arg("-Clto").opt().run();

    let bytes = fs_wrapper::read("foo.wasm");
    println!("{}", bytes.len());
    assert!(bytes.len() < 50_000);
}
