//@ aux-build: issue_67893.rs
//@ edition:2018

extern crate issue_67893;

fn g(_: impl Send) {}

fn main() {
    g(issue_67893::run())
    //~^ ERROR `std::sync::MutexGuard<'_, ()>` cannot be sent between threads safely
}
