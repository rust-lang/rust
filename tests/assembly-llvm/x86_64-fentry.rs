//@ assembly-output: emit-asm
//@ compile-flags: -Zinstrument-fentry=y -Cllvm-args=-x86-asm-syntax=intel

//@ revisions: x86_64-linux
//@[x86_64-linux] compile-flags: --target=x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86
//@[x86_64-linux] only-x86_64-unknown-linux-gnu

#![crate_type = "lib"]

// CHECK-LABEL: mcount_func:
#[no_mangle]
pub fn mcount_func() {
    // CHECK: call __fentry__

    std::hint::black_box(());

    // CHECK: ret
}
