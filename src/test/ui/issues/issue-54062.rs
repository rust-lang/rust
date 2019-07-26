use std::sync::Mutex;

struct Test {
    comps: Mutex<String>,
}

fn main() {}

fn testing(test: Test) {
    let _ = test.comps.inner.lock().unwrap();
    //~^ ERROR: field `inner` of struct `std::sync::Mutex` is private
    //~| ERROR: no method named `unwrap` found
}
