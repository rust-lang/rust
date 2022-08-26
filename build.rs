fn main() {
    // Re-export the TARGET environment variable so it can
    // be accessed by miri.
    let target = std::env::var("TARGET").unwrap();
    println!("cargo:rustc-env=TARGET={}", target);
}
