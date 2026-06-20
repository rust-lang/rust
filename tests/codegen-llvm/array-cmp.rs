// Ensure the asm for array comparisons is properly optimized.

//@ compile-flags: -C opt-level=2
//@ needs-deterministic-layouts (checks depend on tuple layout)

#![crate_type = "lib"]

// CHECK-LABEL: @compare
// CHECK: start:
// CHECK-NEXT: ret i1 true
#[no_mangle]
pub fn compare() -> bool {
    let bytes = 12.5f32.to_ne_bytes();
    bytes
        == if cfg!(target_endian = "big") {
            [0x41, 0x48, 0x00, 0x00]
        } else {
            [0x00, 0x00, 0x48, 0x41]
        }
}

// CHECK-LABEL: @array_of_tuple_le
#[no_mangle]
pub fn array_of_tuple_le(a: &[(i16, u16); 2], b: &[(i16, u16); 2]) -> bool {
    // Ensure that, after all the optimizations have run, the happy path just checks
    // `eq` on each corresponding pair and moves onto the next one if it is.
    // Then there's a dedup'd comparison for the place that's different.
    // (As opposed to, say, running a full `[su]cmp` as part of checking equality.)

    // This is written quite specifically because different library code was triggering
    // <https://github.com/llvm/llvm-project/issues/132678> along the way, so this
    // has enough checks to make sure that's not happening. It doesn't need to be
    // *exactly* this IR, but be careful if you ever need to update these checks.

    // CHECK: start:
    // CHECK: %[[A00:.+]] = load i16, ptr %a
    // CHECK: %[[B00:.+]] = load i16, ptr %b
    // CHECK-NOT: cmp
    // CHECK: %[[EQ00:.+]] = icmp eq i16 %[[A00]], %[[B00]]
    // CHECK-NEXT: br i1 %[[EQ00]], label %[[L01:.+]], label %[[EXIT_S:.+]]

    // CHECK: [[L01]]:
    // CHECK: %[[PA01:.+]] = getelementptr{{.+}}i8, ptr %a, {{i32|i64}} 2
    // CHECK: %[[PB01:.+]] = getelementptr{{.+}}i8, ptr %b, {{i32|i64}} 2
    // CHECK: %[[A01:.+]] = load i16, ptr %[[PA01]]
    // CHECK: %[[B01:.+]] = load i16, ptr %[[PB01]]
    // CHECK-NOT: cmp
    // CHECK: %[[EQ01:.+]] = icmp eq i16 %[[A01]], %[[B01]]
    // CHECK-NEXT: br i1 %[[EQ01]], label %[[L10:.+]], label %[[EXIT_U:.+]]

    // CHECK: [[L10]]:
    // CHECK: %[[PA10:.+]] = getelementptr{{.+}}i8, ptr %a, {{i32|i64}} 4
    // CHECK: %[[PB10:.+]] = getelementptr{{.+}}i8, ptr %b, {{i32|i64}} 4
    // CHECK: %[[A10:.+]] = load i16, ptr %[[PA10]]
    // CHECK: %[[B10:.+]] = load i16, ptr %[[PB10]]
    // CHECK-NOT: cmp
    // CHECK: %[[EQ10:.+]] = icmp eq i16 %[[A10]], %[[B10]]
    // CHECK-NEXT: br i1 %[[EQ10]], label %[[L11:.+]], label %[[EXIT_S]]

    // CHECK: [[L11]]:
    // CHECK: %[[PA11:.+]] = getelementptr{{.+}}i8, ptr %a, {{i32|i64}} 6
    // CHECK: %[[PB11:.+]] = getelementptr{{.+}}i8, ptr %b, {{i32|i64}} 6
    // CHECK: %[[A11:.+]] = load i16, ptr %[[PA11]]
    // CHECK: %[[B11:.+]] = load i16, ptr %[[PB11]]
    // CHECK-NOT: cmp
    // CHECK: %[[EQ11:.+]] = icmp eq i16 %[[A11]], %[[B11]]
    // CHECK-NEXT: br i1 %[[EQ11]], label %[[DONE:.+]], label %[[EXIT_U]]

    // CHECK: [[DONE]]:
    // CHECK: %[[RET:.+]] = phi i1 [ %{{.+}}, %[[EXIT_S]] ], [ %{{.+}}, %[[EXIT_U]] ], [ true, %[[L11]] ]
    // CHECK: ret i1 %[[RET]]

    a <= b
}
