use std::sync::Mutex;

struct Test {
    comps: Mutex<String>,
}

fn main() {}

fn testing(test: Test) {
    let _ = test.comps.inner.try_lock();
    //~^ ERROR: field `inner` of struct `std::sync::Mutex` is private
}

// https://github.com/rust-lang/rust/issues/54062
