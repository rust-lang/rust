#![feature(never_type)]
fn foo() -> Result<u32, !> {
    Ok(123)
}

fn main() {
    let Ok(_x) = foo(); //~ ERROR refutable pattern in local binding
}
