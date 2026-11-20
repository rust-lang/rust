//@ ignore-std-debug-assertions (FIXME: checks for call detect scoped noalias metadata)
//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
#![crate_type = "lib"]

// Ensure that trivial casts of vec elements are O(1)

pub struct Wrapper<T>(T);

// previously repr(C) caused the optimization to fail
#[repr(C)]
pub struct Foo {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

// implementing Copy exercises the TrustedRandomAccess specialization inside the in-place
// specialization
#[derive(Copy, Clone)]
pub struct Bar {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

// this exercises the try-fold codepath
pub struct Baz {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

// CHECK-LABEL: @vec_iterator_cast_primitive
#[no_mangle]
pub fn vec_iterator_cast_primitive(vec: Vec<i8>) -> Vec<u8> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK: call{{.+}}void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| e as u8).collect()
}

// CHECK-LABEL: @vec_iterator_cast_wrapper
#[no_mangle]
pub fn vec_iterator_cast_wrapper(vec: Vec<u8>) -> Vec<Wrapper<u8>> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK: call{{.+}}void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| Wrapper(e)).collect()
}

// CHECK-LABEL: @vec_iterator_cast_signed
#[no_mangle]
pub fn vec_iterator_cast_signed(vec: Vec<i32>) -> Vec<u32> {
    // CHECK-NOT: and i{{[0-9]+}} %{{.*}}, {{[0-9]+}}
    vec.into_iter().map(|e| u32::from_ne_bytes(e.to_ne_bytes())).collect()
}

// CHECK-LABEL: @vec_iterator_cast_signed_nested
#[no_mangle]
pub fn vec_iterator_cast_signed_nested(vec: Vec<Vec<i32>>) -> Vec<Vec<u32>> {
    // CHECK-NOT: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
    // CHECK-NOT: %{{.*}} = udiv
    vec.into_iter()
        .map(|e| e.into_iter().map(|e| u32::from_ne_bytes(e.to_ne_bytes())).collect())
        .collect()
}

// CHECK-LABEL: @vec_iterator_cast_unwrap
#[no_mangle]
pub fn vec_iterator_cast_unwrap(vec: Vec<Wrapper<u8>>) -> Vec<u8> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK: call{{.+}}void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| e.0).collect()
}

// CHECK-LABEL: @vec_iterator_cast_aggregate
#[no_mangle]
pub fn vec_iterator_cast_aggregate(vec: Vec<[u64; 4]>) -> Vec<Foo> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK: call{{.+}}void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}

// CHECK-LABEL: @vec_iterator_cast_deaggregate_tra
#[no_mangle]
pub fn vec_iterator_cast_deaggregate_tra(vec: Vec<Bar>) -> Vec<[u64; 4]> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK: call{{.+}}void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: loop
    // CHECK-NOT: call

    // Safety: For the purpose of this test we assume that Bar layout matches [u64; 4].
    // This currently is not guaranteed for repr(Rust) types, but it happens to work here and
    // the UCG may add additional guarantees for homogenous types in the future that would make this
    // correct.
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}

// CHECK-LABEL: @vec_iterator_cast_deaggregate_fold
#[no_mangle]
pub fn vec_iterator_cast_deaggregate_fold(vec: Vec<Baz>) -> Vec<[u64; 4]> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK: call{{.+}}void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: loop
    // CHECK-NOT: call

    // Safety: For the purpose of this test we assume that Bar layout matches [u64; 4].
    // This currently is not guaranteed for repr(Rust) types, but it happens to work here and
    // the UCG may add additional guarantees for homogenous types in the future that would make this
    // correct.
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}

// CHECK-LABEL: @vec_iterator_cast_unwrap_drop
#[no_mangle]
pub fn vec_iterator_cast_unwrap_drop(vec: Vec<Wrapper<String>>) -> Vec<String> {
    // CHECK-NOT: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
    // CHECK-NOT: %{{.*}} = mul
    // CHECK-NOT: %{{.*}} = udiv
    // CHECK: call
    // CHECK-SAME: void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
    // CHECK-NOT: call
    // CHECK-NOT: %{{.*}} = mul
    // CHECK-NOT: %{{.*}} = udiv
    // CHECK: ret void

    vec.into_iter().map(|Wrapper(e)| e).collect()
}

// CHECK-LABEL: @vec_iterator_cast_wrap_drop
#[no_mangle]
pub fn vec_iterator_cast_wrap_drop(vec: Vec<String>) -> Vec<Wrapper<String>> {
    // CHECK-NOT: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
    // CHECK-NOT: %{{.*}} = mul
    // CHECK-NOT: %{{.*}} = udiv
    // CHECK: call
    // CHECK-SAME: void @llvm.assume(i1 %{{.+}})
    // CHECK-NOT: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
    // CHECK-NOT: call
    // CHECK-NOT: %{{.*}} = mul
    // CHECK-NOT: %{{.*}} = udiv
    // CHECK: ret void

    vec.into_iter().map(Wrapper).collect()
}
