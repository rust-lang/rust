// ignore-macos
// ignore-windows

#![feature(main)]

#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
#[main]
fn a() {
    println!("Hello, World!");
    a();
}
