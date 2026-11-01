//@ edition:2021

// Test that array access is not stored as part of closure kind origin

fn expect_fn<F: Fn()>(_f: F) {}

fn main() {
    let s = [format!("s"), format!("s")];
    let c = || { //~ ERROR expected a closure that implements the `Fn`
        let [_, _s] = s;
    };
    expect_fn(c);
}
