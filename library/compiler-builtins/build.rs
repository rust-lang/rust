use std::env;

fn main() {
    if env::var("TARGET").unwrap().ends_with("gnueabihf") {
        println!("cargo:rustc-cfg=gnueabihf")
    }
}
