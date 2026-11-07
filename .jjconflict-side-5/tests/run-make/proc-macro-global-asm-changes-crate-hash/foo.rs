// Consumer crate. Byte-identical across all invocations of the test;
// only the asm template inside the `global_asm!` block spliced in by
// `changing_macro::emit_global_asm!` differs between builds.

#![crate_type = "rlib"]

extern crate changing_macro;

changing_macro::emit_global_asm!();
