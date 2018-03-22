use std::env;

fn main() {
    println!(
        "cargo:rustc-env=TARGET={}",
        env::var("TARGET").unwrap()
    );
}
