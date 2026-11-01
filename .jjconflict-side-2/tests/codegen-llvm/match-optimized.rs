//@ compile-flags: -Cno-prepopulate-passes -Copt-level=3

#![crate_type = "lib"]

pub enum E {
    A,
    B,
    C,
}

// CHECK-LABEL: @exhaustive_match
#[no_mangle]
pub fn exhaustive_match(e: E) -> u8 {
    // CHECK: switch{{.*}}, label %[[OTHERWISE:[a-zA-Z0-9_]+]] [
    // CHECK-NEXT: i[[TY:[0-9]+]] [[DISCR:[0-9]+]], label %[[A:[a-zA-Z0-9_]+]]
    // CHECK-NEXT: i[[TY:[0-9]+]] [[DISCR:[0-9]+]], label %[[B:[a-zA-Z0-9_]+]]
    // CHECK-NEXT: i[[TY:[0-9]+]] [[DISCR:[0-9]+]], label %[[C:[a-zA-Z0-9_]+]]
    // CHECK-NEXT: ]
    // CHECK: [[OTHERWISE]]:
    // CHECK-NEXT: unreachable
    //
    // CHECK: [[A]]:
    // CHECK-NEXT: store i8 0, ptr %_0, align 1
    // CHECK-NEXT: br label %[[EXIT:[a-zA-Z0-9_]+]]
    // CHECK: [[B]]:
    // CHECK-NEXT: store i8 1, ptr %_0, align 1
    // CHECK-NEXT: br label %[[EXIT]]
    // CHECK: [[C]]:
    // CHECK-NEXT: store i8 3, ptr %_0, align 1
    // CHECK-NEXT: br label %[[EXIT]]
    match e {
        E::A => 0,
        E::B => 1,
        E::C => 3,
    }
}

#[repr(u16)]
pub enum E2 {
    A = 13,
    B = 42,
}

// For optimized code we produce a switch with an unreachable target as the `otherwise` so LLVM
// knows the possible values. Compare with `tests/codegen/match-unoptimized.rs`.

// CHECK-LABEL: @exhaustive_match_2
#[no_mangle]
pub fn exhaustive_match_2(e: E2) -> u8 {
    // CHECK: switch i16 %{{.+}}, label %[[UNREACH:.+]] [
    // CHECK-NEXT: i16 13,
    // CHECK-NEXT: i16 42,
    // CHECK-NEXT: ]
    // CHECK: [[UNREACH]]:
    // CHECK-NEXT: unreachable
    match e {
        E2::A => 0,
        E2::B => 1,
    }
}
