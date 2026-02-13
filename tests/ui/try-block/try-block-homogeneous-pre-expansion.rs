//@ check-pass
//@ edition: 2018

// For historical reasons this is only a warning, not an error.
// See <https://github.com/rust-lang/rust/issues/152501>

fn main() {
    #[cfg(false)]
    try {}
    //~^ warn `try` blocks are unstable
    //~| warn unstable syntax can change at any point
}
