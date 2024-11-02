// Verifies that setting the tls-dialect result sin the correct set of assembly
// instructions.
//
// Note: tls-dialect flags have no changes to LLVM IR, and only affect which
// instruction sequences are emitted by the LLVM backend. Checking the assembly
// output is how we test the lowering in LLVM, and is the only way a frontend
// can determine if its code generation flags are set correctly.
//
//@ revisions: x64 x64-trad x64-desc
//
//@[x64]      compile-flags: --target=x86_64-unknown-linux-gnu
//@[x64-trad] compile-flags: --target=x86_64-unknown-linux-gnu -Z tls-dialect=trad
//@[x64-desc] compile-flags: --target=x86_64-unknown-linux-gnu -Z tls-dialect=desc
//
//@ assembly-output: emit-asm
//@ aux-build:tlsdesc_aux.rs

#![crate_type = "lib"]

extern crate tlsdesc_aux as aux;

#[no_mangle]
fn get_aux() -> u64 {
    // x64:      __tls_get_addr
    // x64-trad: __tls_get_addr
    // x64-desc: tlsdesc
    // x64-desc: tlscall
    aux::A.with(|a| a.get())
}
