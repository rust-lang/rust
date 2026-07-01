//@ compile-flags: -O
//@ only-64bit

#![crate_type = "lib"]

// CHECK-LABEL: @resize_bytes_is_one_memset
#[no_mangle]
pub fn resize_bytes_is_one_memset(x: &mut Vec<u8>) {
    // CHECK: call void @llvm.memset.p0.i64({{.+}}, i8 123, i64 456789, i1 false)
    let new_len = x.len() + 456789;
    x.resize(new_len, 123);
}

#[derive(Copy, Clone)]
struct ByteNewtype(i8);

// CHECK-LABEL: @from_elem_is_one_memset
#[no_mangle]
pub fn from_elem_is_one_memset() -> Vec<ByteNewtype> {
    // CHECK: %[[P:.+]] = tail call{{.+}}@__rust_alloc(i64 noundef 123456, i64 noundef 1)
    // CHECK: call void @llvm.memset.p0.i64({{.+}} %[[P]], i8 42, i64 123456, i1 false)
    vec![ByteNewtype(42); 123456]
}
