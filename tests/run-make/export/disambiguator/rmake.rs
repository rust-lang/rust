use run_make_support::rustc;

fn main() {
    rustc().input("libr.rs").arg("--emit=i-rs").run();
    rustc()
        .env("RUSTC_FORCE_RUSTC_VERSION", "1")
        .input("libr.rs")
        .arg("-Csymbol-mangling-version=v0")
        .run();
    rustc()
        .env("RUSTC_FORCE_RUSTC_VERSION", "2")
        .input("app.rs")
        .arg("-Csymbol-mangling-version=v0")
        .run();
}
