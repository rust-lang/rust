fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_BOOTSTRAP");
    if std::env::var("RUSTC_BOOTSTRAP").is_err() {
        eprintln!(
            "error: you are attempting to build the compiler without going through bootstrap"
        );
        eprintln!(
            "help: see https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html for how to build the compiler"
        );
        eprintln!(
            "help: if you know what you're doing, set the RUSTC_BOOTSTRAP environment variable to any value"
        );
        panic!("wrong command used for building");
    }
}
