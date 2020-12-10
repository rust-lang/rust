// compile-flags: -O -C panic=abort

#![crate_type = "lib"]
#![feature(extend_from_array)]

// This test is here to ensure that LLVM is optimizing out the `ManuallyDrop` construction,
// since we really don't want to copy the whole array to the stack and then again to the Vec.

// CHECK-LABEL: @vec_extend_from_array_demo
#[no_mangle]
pub fn vec_extend_from_array_demo(v: &mut Vec<String>, a: [String; 400]) {
    // CHECK-NOT: alloca
    // CHECK: call alloc::vec::Vec<T,A>::reserve
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy
    // CHECK-NOT: alloca
    // CHECK: ret
    v.extend_from_array(a);
}

// No validation against this one; it just keeps `reserve` from having only a single caller.
pub fn please_do_not_inline_the_reserve_call_llvm(v: &mut Vec<String>, s: String) {
    v.push(s);
}
