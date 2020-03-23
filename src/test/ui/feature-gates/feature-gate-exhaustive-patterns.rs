// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

#![feature(never_type)]

fn foo() -> Result<u32, !> {
    Ok(123)
}

fn main() {
    let Ok(_x) = foo(); //~ ERROR refutable pattern in local binding
}
