// edition:2018

// tests that editions work with the tyvar warning-turned-error

#[deny(warnings)]
fn main() {
    let x = 0;
    let y = &x as *const _;
    let _ = y.is_null();
    //~^ error: the type of this value must be known to call a method on a raw pointer on it [E0699]
}
