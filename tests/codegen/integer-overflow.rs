//@ compile-flags: -Copt-level=3 -C overflow-checks=on

#![crate_type = "lib"]

pub struct S1<'a> {
    data: &'a [u8],
    position: usize,
}

// CHECK-LABEL: @slice_no_index_order
#[no_mangle]
pub fn slice_no_index_order<'a>(s: &'a mut S1, n: usize) -> &'a [u8] {
    // CHECK-NOT: slice_index_order_fail
    let d = &s.data[s.position..s.position + n];
    s.position += n;
    return d;
}

// CHECK-LABEL: @test_check
#[no_mangle]
pub fn test_check<'a>(s: &'a mut S1, x: usize, y: usize) -> &'a [u8] {
    // CHECK: slice_index_order_fail
    &s.data[x..y]
}
