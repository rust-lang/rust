//@ compile-flags: -O

#![crate_type = "lib"]

// Tests that codegen works properly when enums like `Result<usize, Box<()>>`
// are represented as `{ u64, ptr }`, i.e., for `Ok(123)`, `123` is stored
// as a pointer.

// CHECK-LABEL: @insert_int
#[no_mangle]
pub fn insert_int(x: usize) -> Result<usize, Box<()>> {
    // CHECK: start:
    // CHECK-NEXT: inttoptr i{{[0-9]+}} %x to ptr
    // CHECK-NEXT: insertvalue
    // CHECK-NEXT: ret { i{{[0-9]+}}, ptr }
    Ok(x)
}

// CHECK-LABEL: @insert_box
#[no_mangle]
pub fn insert_box(x: Box<()>) -> Result<usize, Box<()>> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i{{[0-9]+}}, ptr }
    // CHECK-NEXT: ret
    Err(x)
}

// CHECK-LABEL: @extract_int
#[no_mangle]
pub unsafe fn extract_int(x: Result<usize, Box<()>>) -> usize {
    // CHECK: ptrtoint
    x.unwrap_unchecked()
}

// CHECK-LABEL: @extract_box
#[no_mangle]
pub unsafe fn extract_box(x: Result<usize, Box<()>>) -> Box<()> {
    // CHECK-NOT: ptrtoint
    // CHECK-NOT: inttoptr
    // CHECK-NOT: load
    // CHECK-NOT: store
    x.unwrap_err_unchecked()
}
