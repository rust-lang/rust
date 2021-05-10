// aux-build: issue_67893.rs
// edition:2018
// compile-flags: --error-format human-annotate-rs

extern crate issue_67893;

fn g(_: impl Send) {}

fn main() {
    g(issue_67893::run())
    //~^ ERROR: `MutexGuard<'_, ()>` cannot be sent between threads safely
}
