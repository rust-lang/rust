//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// Adapted from <https://github.com/rust-lang/rust/issues/73258#issue-637346014>

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Foo {
    A,
    B,
    C,
    D,
}

// CHECK-LABEL: @issue_73258(
#[no_mangle]
pub unsafe fn issue_73258(ptr: *const Foo) -> Foo {
    // CHECK-NOT: icmp
    // CHECK-NOT: call
    // CHECK-NOT: br
    // CHECK-NOT: select

    // CHECK: %[[R:.+]] = load i8
    // CHECK-SAME: !range !

    // CHECK-NOT: icmp
    // CHECK-NOT: call
    // CHECK-NOT: br
    // CHECK-NOT: select

    // CHECK: ret i8 %[[R]]

    // CHECK-NOT: icmp
    // CHECK-NOT: call
    // CHECK-NOT: br
    // CHECK-NOT: select
    let k: Option<Foo> = Some(ptr.read());
    return k.unwrap();
}
