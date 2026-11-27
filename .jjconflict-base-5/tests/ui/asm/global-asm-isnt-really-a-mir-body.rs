//@ revisions: emit_mir instrument cfi

//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer

// Make sure we don't try to emit MIR for it.
//@[emit_mir] compile-flags: --emit=mir

// Make sure we don't try to instrument it.
//@[instrument] compile-flags: -Cinstrument-coverage -Zno-profiler-runtime
//@[instrument] only-linux

// Make sure we don't try to CFI encode it.
//@[cfi] compile-flags: -Zsanitizer=cfi -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Clink-dead-code=true
//@[cfi] needs-sanitizer-cfi
//@[cfi] no-prefer-dynamic
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@[cfi] only-linux

//@ build-pass
//@ needs-asm-support
//@ ignore-backends: gcc

use std::arch::global_asm;

fn foo() {}

global_asm!("/* {} */", sym foo);

fn main() {}
