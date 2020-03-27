// Test LVI load hardening on SGX module level assembly code

// assembly-output: emit-asm
// compile-flags: --crate-type staticlib
// only-x86_64-fortanix-unknown-sgx

#![feature(global_asm)]

global_asm!(".start_module_asm:
            movq (%rdi), %rax
            retq
            .end_module_asm:" );
