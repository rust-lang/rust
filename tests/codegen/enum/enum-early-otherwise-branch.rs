//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

pub enum Enum {
    A(u32),
    B(u32),
    C(u32),
}

#[no_mangle]
pub fn foo(lhs: &Enum, rhs: &Enum) -> bool {
    // CHECK-LABEL: define{{.*}}i1 @foo(
    // CHECK-NOT: switch
    // CHECK-NOT: br
    // CHECK: [[SELECT:%.*]] = select
    // CHECK-NEXT: ret i1 [[SELECT]]
    // CHECK-NEXT: }
    match (lhs, rhs) {
        (Enum::A(lhs), Enum::A(rhs)) => lhs == rhs,
        (Enum::B(lhs), Enum::B(rhs)) => lhs == rhs,
        (Enum::C(lhs), Enum::C(rhs)) => lhs == rhs,
        _ => false,
    }
}
