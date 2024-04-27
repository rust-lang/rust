//@ignore-target-apple

#![feature(rustc_attrs)]

#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
#[rustc_main]
fn a() {
    println!("Hello, World!");
    a();
}
