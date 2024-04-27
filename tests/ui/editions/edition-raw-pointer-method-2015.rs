//@ edition:2015

// tests that editions work with the tyvar warning-turned-error

#[deny(warnings)]
fn main() {
    let x = 0;
    let y = &x as *const _;
    let _ = y.is_null();
    //~^ error: type annotations needed [tyvar_behind_raw_pointer]
    //~^^ warning: this is accepted in the current edition
}
