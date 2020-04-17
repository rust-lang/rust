// aux-build: issue_67893.rs
// edition:2018
// dont-check-compiler-stderr
// FIXME(#71222): Add above flag because of the difference of stderrs on some env.

extern crate issue_67893;

fn g(_: impl Send) {}

fn main() {
    g(issue_67893::run())
    //~^ ERROR: `std::sync::MutexGuard<'_, ()>` cannot be sent between threads safely
}
