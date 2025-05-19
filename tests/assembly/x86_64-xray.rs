//@ assembly-output: emit-asm
//@ compile-flags: --crate-type lib -Zinstrument-xray=always -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx

// CHECK-LABEL: xray_func:
#[no_mangle]
pub fn xray_func() {
    // CHECK: nop word ptr [rax + rax + 512]

    std::hint::black_box(());

    // CHECK: ret
    // CHECK-NEXT: nop word ptr cs:[rax + rax + 512]
}
