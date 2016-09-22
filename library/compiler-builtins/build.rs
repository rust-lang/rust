use std::env;

fn main() {
    if env::var("TARGET").unwrap().ends_with("hf") {
        println!("cargo:rustc-cfg=gnueabihf")
    }
}
