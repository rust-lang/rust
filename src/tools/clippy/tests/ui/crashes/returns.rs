//@ check-pass
// Test for https://github.com/rust-lang/rust-clippy/issues/1346

#[deny(warnings)]
fn cfg_return() -> i32 {
    #[cfg(unix)]
    return 1;
    #[cfg(not(unix))]
    return 2;
}

#[deny(warnings)]
fn cfg_let_and_return() -> i32 {
    #[cfg(unix)]
    let x = 1;
    #[cfg(not(unix))]
    let x = 2;
    x
}

fn main() {
    cfg_return();
    cfg_let_and_return();
}
