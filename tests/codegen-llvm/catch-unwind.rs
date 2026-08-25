//@ compile-flags: -Copt-level=3

// On x86 the closure is inlined in foo() producing something like
// define i32 @foo() [...] {
// tail call void @bar() [...]
// ret i32 0
// }
// On riscv the closure is another function, placed before fn foo so CHECK can't
// find it
//@ ignore-riscv64 FIXME
// On s390x the closure is also in another function
//@ ignore-s390x FIXME
// On loongarch64 the closure is also in another function
//@ ignore-loongarch64 FIXME

#![crate_type = "lib"]

extern "C" {
    fn bar();
}

// CHECK-LABEL: @foo
#[no_mangle]
pub unsafe fn foo() -> i32 {
    // CHECK: call void @bar
    // CHECK: ret i32 0
    std::panic::catch_unwind(|| {
        bar();
        0
    })
    .unwrap()
}
