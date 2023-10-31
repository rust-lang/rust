// edition:2018

// tests that editions work with the tyvar warning-turned-error

#[deny(warnings)]
fn main() {
    let x = 0;
    let y = &x as *const _;
    let _ = y.is_null();
    //~^ error: cannot call a method on a raw pointer with an unknown pointee type [E0699]
}
