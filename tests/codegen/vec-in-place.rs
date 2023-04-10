// ignore-debug: the debug assertions get in the way
// compile-flags: -O -Z merge-functions=disabled
// min-llvm-version: 16
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
    vec.into_iter().map(|e| e as u8).collect()
}

// CHECK-LABEL: @vec_iterator_cast_wrapper
#[no_mangle]
pub fn vec_iterator_cast_wrapper(vec: Vec<u8>) -> Vec<Wrapper<u8>> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| Wrapper(e)).collect()
}

// CHECK-LABEL: @vec_iterator_cast_unwrap
#[no_mangle]
pub fn vec_iterator_cast_unwrap(vec: Vec<Wrapper<u8>>) -> Vec<u8> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| e.0).collect()
}

// CHECK-LABEL: @vec_iterator_cast_aggregate
#[no_mangle]
pub fn vec_iterator_cast_aggregate(vec: Vec<[u64; 4]>) -> Vec<Foo> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}

// CHECK-LABEL: @vec_iterator_cast_deaggregate_tra
#[no_mangle]
pub fn vec_iterator_cast_deaggregate_tra(vec: Vec<Bar>) -> Vec<[u64; 4]> {
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

    // Safety: For the purpose of this test we assume that Bar layout matches [u64; 4].
    // This currently is not guaranteed for repr(Rust) types, but it happens to work here and
    // the UCG may add additional guarantees for homogenous types in the future that would make this
    // correct.
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}
