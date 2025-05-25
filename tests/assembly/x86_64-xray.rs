//@ assembly-output: emit-asm
//@ compile-flags: -Zinstrument-xray=always -C llvm-args=-x86-asm-syntax=intel --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ only-linux

#![crate_type = "lib"]

// CHECK-LABEL: xray_func:
#[no_mangle]
pub fn xray_func() {
    // CHECK: nop word ptr [rax + rax + 512]

    std::hint::black_box(());

    // CHECK: ret
    // CHECK-NEXT: nop word ptr cs:[rax + rax + 512]
}
