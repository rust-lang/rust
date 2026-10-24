//! `main` is just a wrapper to handle configuration.

#[cfg(not(feature = "build-mpfr"))]
fn main() {
    eprintln!("multiprecision not enabled; skipping extensive tests");
}

#[cfg(feature = "build-mpfr")]
mod run;

#[cfg(feature = "build-mpfr")]
fn main() {
    run::run();
}
