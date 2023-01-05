// edition:2018
// gate-test-async_closure

fn f() {
    let _ = async || {}; //~ ERROR async closures are unstable
}

fn main() {}
