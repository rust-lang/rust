//@ assembly-output: emit-asm
//@ revisions: att intel
//@ [att] compile-flags: -Cllvm-args=-x86-asm-syntax=att
//@ [intel] compile-flags: -Cllvm-args=-x86-asm-syntax=intel
//@ only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: naked_att:
// intel-CHECK: mov rax, qword ptr [rdi]
// intel-CHECK: ret
// att-CHECK: movq (%rdi), %rax
// att-CHECK: retq

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "sysv64" fn naked_att() {
    std::arch::naked_asm!(
        "
        movq (%rdi), %rax
        retq
        ",
        options(att_syntax),
    );
}

// CHECK-LABEL: naked_intel:
// intel-CHECK: mov rax, rdi
// intel-CHECK: ret
// att-CHECK: movq (%rdi), %rax
// att-CHECK: retq

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "sysv64" fn naked_intel() {
    std::arch::naked_asm!(
        "
        mov rax, rdi
        ret
        ",
        options(),
    );
}

// CHECK-LABEL: global_att:
// intel-CHECK: mov rax, rdi
// intel-CHECK: ret
// att-CHECK: movq (%rdi), %rax
// att-CHECK: retq

core::arch::global_asm!(
    "
    .globl global_att
    global_att:
        movq (%rdi), %rax
        retq
    ",
    options(att_syntax),
);

// CHECK-LABEL: global_intel:
// intel-CHECK: mov rax, rdi
// intel-CHECK: ret
// att-CHECK: movq (%rdi), %rax
// att-CHECK: retq

core::arch::global_asm!(
    "
    .globl global_intel
    global_intel:
        mov rax, rdi
        ret
    ",
    options(),
);
