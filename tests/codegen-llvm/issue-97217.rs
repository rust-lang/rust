//@ compile-flags: -C opt-level=3
#![crate_type = "lib"]

// Regression test for issue 97217 (the following should result in no allocations)

// CHECK-LABEL: @issue97217
#[no_mangle]
pub fn issue97217() -> i32 {
    // drop_in_place should be inlined and never appear
    // CHECK-NOT: drop_in_place

    // __rust_alloc should be optimized out
    // CHECK-NOT: __rust_alloc

    let v1 = vec![5, 6, 7];
    let v1_iter = v1.iter();
    let total: i32 = v1_iter.sum();
    println!("{}", total);
    total
}
