// compile-flags: -O

// On x86 the closure is inlined in foo() producting something like
// define i32 @foo() [...] {
// tail call void @bar() [...]
// ret i32 0
// }
// On riscv the closure is another function, placed before fn foo so CHECK can't
// find it
// ignore-riscv64 FIXME

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
