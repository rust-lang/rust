//! `main` is just a wrapper to handle configuration.

#[cfg(not(feature = "test-multiprecision"))]
fn main() {
    eprintln!("multiprecision not enabled; skipping extensive tests");
}

#[cfg(feature = "test-multiprecision")]
mod run;

#[cfg(feature = "test-multiprecision")]
fn main() {
    run::run();
}
