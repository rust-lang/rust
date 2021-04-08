// ignore-macos
// ignore-windows

#![feature(rustc_main)]
#![rustc_main(a)]

#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
fn a() {
    println!("Hello, World!");
    a();
}
