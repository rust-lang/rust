// ignore-windows
// ignore-freebsd
// ignore-openbsd
// ignore-netbsd
// ignore-bitrig

// compile-flags: -Z parse-only

#[path = "../compile-fail"]
mod foo; //~ ERROR: a directory

fn main() {}
