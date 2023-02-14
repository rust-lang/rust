// ignore-debug: the debug assertions get in the way
// compile-flags: -O -Z merge-functions=disabled
#![crate_type = "lib"]

// Ensure that trivial casts of vec elements are O(1)

pub struct Wrapper<T>(T);

#[repr(C)]
pub struct Foo {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

// Going from an aggregate struct to another type currently requires Copy to
// enable the TrustedRandomAccess specialization. Without it optimizations do not yet
// reliably recognize the loops as noop for repr(C) or non-Copy structs.
#[derive(Copy, Clone)]
pub struct Bar {
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
    // FIXME These checks should be the same as other functions.
    // CHECK-NOT: @__rust_alloc
    // CHECK-NOT: @__rust_alloc
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}

// CHECK-LABEL: @vec_iterator_cast_deaggregate
#[no_mangle]
pub fn vec_iterator_cast_deaggregate(vec: Vec<Bar>) -> Vec<[u64; 4]> {
    // FIXME These checks should be the same as other functions.
    // CHECK-NOT: @__rust_alloc
    // CHECK-NOT: @__rust_alloc

    // Safety: For the purpose of this test we assume that Bar layout matches [u64; 4].
    // This currently is not guaranteed for repr(Rust) types, but it happens to work here and
    // the UCG may add additional guarantees for homogenous types in the future that would make this
    // correct.
    vec.into_iter().map(|e| unsafe { std::mem::transmute(e) }).collect()
}
