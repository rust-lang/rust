//@ edition:2018

// tests that editions work with the tyvar warning-turned-error

#[deny(warnings)]
fn main() {
    let x = 0;
    let y = &x as *const _;
    //~^ error: type annotations needed
    let _ = y.is_null();
}
