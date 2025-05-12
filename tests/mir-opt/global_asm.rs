// skip-filecheck
//@ needs-asm-support

// `global_asm!` gets a fake body, make sure it is handled correctly

// EMIT_MIR global_asm.{global_asm#0}.SimplifyLocals-final.after.mir
core::arch::global_asm!("/* */");

fn main() {}
