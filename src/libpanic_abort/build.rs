fn main() {
    // Hack to force this crate to be compiled with the `abort`
    // panic strategy, regardless of what strategy Cargo
    // passes to the compiler.
    // See `rustc::session::Session::panic_strategy` for more details
    println!("cargo:rustc-env=RUSTC_INTERNAL_FORCE_PANIC_ABORT=1");
}
