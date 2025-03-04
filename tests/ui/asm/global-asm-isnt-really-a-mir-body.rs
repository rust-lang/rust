//@ revisions: emit_mir instrument cfi

// Make sure we don't try to emit MIR for it.
//@[emit_mir] compile-flags: --emit=mir

// Make sure we don't try to instrument it.
//@[instrument] compile-flags: -Cinstrument-coverage -Zno-profiler-runtime
//@[instrument] only-linux

// Make sure we don't try to CFI encode it.
//@[cfi] compile-flags: -Zsanitizer=cfi -Ccodegen-units=1 -Clto -Clink-dead-code=true
//@[cfi] needs-sanitizer-cfi
//@[cfi] no-prefer-dynamic

//@ build-pass

use std::arch::global_asm;

fn foo() {}

global_asm!("/* {} */", sym foo);

fn main() {}
