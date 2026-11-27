//@ compile-flags: -Copt-level=3 -C overflow-checks=on

#![crate_type = "lib"]

pub struct S1<'a> {
    data: &'a [u8],
    position: usize,
}

// CHECK-LABEL: @slice_no_index_order
#[no_mangle]
pub fn slice_no_index_order<'a>(s: &'a mut S1, n: usize) -> &'a [u8] {
    // CHECK-COUNT-1: slice_index_fail
    let d = &s.data[s.position..s.position + n];
    s.position += n;
    return d;
}

// CHECK-LABEL: @test_check
#[no_mangle]
pub fn test_check<'a>(s: &'a mut S1, x: usize, y: usize) -> &'a [u8] {
    // CHECK-COUNT-1: slice_index_fail
    &s.data[x..y]
}
