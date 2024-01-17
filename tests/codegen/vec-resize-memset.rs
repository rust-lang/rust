// compile-flags: -O
// no-system-llvm
// only-64bit
// ignore-debug (the extra assertions get in the way)

#![crate_type = "lib"]

pub fn resize_with_bytes_is_one_memset(x: &mut Vec<u8>) {
    let new_len = x.len() + 456789;
    x.resize(new_len, 123);
}

// CHECK: call void @llvm.memset.p0.i64({{.+}}, i8 123, i64 456789, i1 false)
