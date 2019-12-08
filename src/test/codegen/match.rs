// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

pub enum E {
    A,
    B,
}

// CHECK-LABEL: @exhaustive_match
#[no_mangle]
pub fn exhaustive_match(e: E) -> u8 {
// CHECK: switch{{.*}}, label %[[OTHERWISE:[a-zA-Z0-9_]+]] [
// CHECK-NEXT: i[[TY:[0-9]+]] [[DISCR:[0-9]+]], label %[[A:[a-zA-Z0-9_]+]]
// CHECK-NEXT: i[[TY:[0-9]+]] [[DISCR:[0-9]+]], label %[[B:[a-zA-Z0-9_]+]]
// CHECK-NEXT: ]
// CHECK: [[B]]:
// CHECK-NEXT: store i8 1, i8* %1, align 1
// CHECK-NEXT: br label %[[EXIT:[a-zA-Z0-9_]+]]
// CHECK: [[OTHERWISE]]:
// CHECK-NEXT: unreachable
// CHECK: [[A]]:
// CHECK-NEXT: store i8 0, i8* %1, align 1
// CHECK-NEXT: br label %[[EXIT:[a-zA-Z0-9_]+]]
    match e {
        E::A => 0,
        E::B => 1,
    }
}
