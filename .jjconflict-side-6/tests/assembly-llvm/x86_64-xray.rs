//@ assembly-output: emit-asm
//@ compile-flags: -Zinstrument-xray=always -Cllvm-args=-x86-asm-syntax=intel

//@ revisions: x86_64-linux
//@[x86_64-linux] compile-flags: --target=x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86
//@[x86_64-linux] only-x86_64-unknown-linux-gnu

//@ revisions: x86_64-darwin
//@[x86_64-darwin] compile-flags: --target=x86_64-apple-darwin
//@[x86_64-darwin] needs-llvm-components: x86
//@[x86_64-darwin] only-x86_64-apple-darwin

#![crate_type = "lib"]

// CHECK-LABEL: xray_func:
#[no_mangle]
pub fn xray_func() {
    // CHECK: nop word ptr [rax + rax + 512]

    std::hint::black_box(());

    // CHECK: ret
    // CHECK-NEXT: nop word ptr cs:[rax + rax + 512]
}
