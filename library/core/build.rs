fn main() {
    if !std::env::var("RUSTC_BOOTSTRAP").is_ok() {
        eprintln!("error: you are attempting to build libcore without going through bootstrap");
        eprintln!("help: use `x.py build --stage 0 library/std`, not `cargo build`");
        eprintln!("help: use `x.py check`, not `cargo check`");
        eprintln!(
            "note: if you're sure you want to do this, use `RUSTC_BOOTSTRAP=0` or some other dummy value"
        );
        panic!();
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP");
}
