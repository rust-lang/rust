// build-pass

core::arch::global_asm!("/* {} */", sym <&'static ()>::clone);

fn main() {}
