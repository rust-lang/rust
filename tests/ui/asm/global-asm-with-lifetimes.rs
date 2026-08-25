//@ build-pass
//@ needs-asm-support

fn foo<T>() {}

core::arch::global_asm!("/* {} */", sym foo::<&'static ()>);

fn main() {}
