//@ build-pass

fn foo<T>() {}

core::arch::global_asm!("/* {} */", sym foo::<&'static ()>);

fn main() {}
