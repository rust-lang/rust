//@ revisions: normal llvm21
//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//@ [llvm21] min-llvm-version: 21
//@ only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: @vec_zero_bytes
#[no_mangle]
pub fn vec_zero_bytes(n: usize) -> Vec<u8> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(
    // CHECK-NOT: call {{.*}}llvm.memset

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(
    // CHECK-NOT: call {{.*}}llvm.memset

    // CHECK: ret void
    vec![0; n]
}

// CHECK-LABEL: @vec_one_bytes
#[no_mangle]
pub fn vec_one_bytes(n: usize) -> Vec<u8> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc_zeroed(

    // CHECK: call {{.*}}__rust_alloc(
    // CHECK: call {{.*}}llvm.memset

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc_zeroed(

    // CHECK: ret void
    vec![1; n]
}

// CHECK-LABEL: @vec_zero_scalar
#[no_mangle]
pub fn vec_zero_scalar(n: usize) -> Vec<i32> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: ret void
    vec![0; n]
}

// CHECK-LABEL: @vec_one_scalar
#[no_mangle]
pub fn vec_one_scalar(n: usize) -> Vec<i32> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc_zeroed(

    // CHECK: call {{.*}}__rust_alloc(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc_zeroed(

    // CHECK: ret void
    vec![1; n]
}

// CHECK-LABEL: @vec_zero_rgb48
#[no_mangle]
pub fn vec_zero_rgb48(n: usize) -> Vec<[u16; 3]> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: ret void
    vec![[0, 0, 0]; n]
}

// CHECK-LABEL: @vec_zero_array_16
#[no_mangle]
pub fn vec_zero_array_16(n: usize) -> Vec<[i64; 16]> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: ret void
    vec![[0_i64; 16]; n]
}

// CHECK-LABEL: @vec_zero_tuple
#[no_mangle]
pub fn vec_zero_tuple(n: usize) -> Vec<(i16, u8, char)> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: ret void
    vec![(0, 0, '\0'); n]
}

// CHECK-LABEL: @vec_non_zero_tuple
#[no_mangle]
pub fn vec_non_zero_tuple(n: usize) -> Vec<(i16, u8, char)> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc_zeroed(

    // CHECK: call {{.*}}__rust_alloc(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc_zeroed(

    // CHECK: ret void
    vec![(0, 0, 'A'); n]
}

// CHECK-LABEL: @vec_option_bool
#[no_mangle]
pub fn vec_option_bool(n: usize) -> Vec<Option<bool>> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: ret void
    vec![Some(false); n]
}

// CHECK-LABEL: @vec_option_i32
#[no_mangle]
pub fn vec_option_i32(n: usize) -> Vec<Option<i32>> {
    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: call {{.*}}__rust_alloc_zeroed(

    // CHECK-NOT: call {{.*}}alloc::vec::from_elem
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}__rust_alloc(

    // CHECK: ret void
    vec![None; n]
}

// LLVM21-LABEL: @vec_array
#[cfg(llvm21)]
#[no_mangle]
pub fn vec_array(n: usize) -> Vec<[u32; 1_000_000]> {
    // LLVM21-NOT: call {{.*}}alloc::vec::from_elem
    // LLVM21-NOT: call {{.*}}reserve
    // LLVM21-NOT: call {{.*}}__rust_alloc(

    // LLVM21: call {{.*}}__rust_alloc_zeroed(

    // LLVM21-NOT: call {{.*}}alloc::vec::from_elem
    // LLVM21-NOT: call {{.*}}reserve
    // LLVM21-NOT: call {{.*}}__rust_alloc(

    // LLVM21: ret void
    vec![[0; 1_000_000]; 3]
}

// Ensure that __rust_alloc_zeroed gets the right attributes for LLVM to optimize it away.
// CHECK: declare noalias noundef ptr @{{.*}}__rust_alloc_zeroed(i64 noundef, i64 allocalign noundef) unnamed_addr [[RUST_ALLOC_ZEROED_ATTRS:#[0-9]+]]

// CHECK-DAG: attributes [[RUST_ALLOC_ZEROED_ATTRS]] = { {{.*}} allockind("alloc,zeroed,aligned") allocsize(0) uwtable "alloc-family"="__rust_alloc" {{.*}} }
