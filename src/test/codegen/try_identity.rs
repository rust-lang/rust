// compile-flags: -O --edition=2021 -Z merge-functions=disabled

// Ensure that common patterns for identity matching results have no overhead.

#![feature(try_blocks)]

#![crate_type = "lib"]

#[no_mangle]
pub fn result_nop_match(x: Result<i32, u32>) -> Result<i32, u32> {
// CHECK: start:
// CHECK-NEXT: ret i64 %0
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

#[no_mangle]
pub fn result_nop_try_block(x: Result<i32, u32>) -> Result<i32, u32> {
// CHECK: start:
// CHECK-NEXT: ret i64 %0
    try {
        x?
    }
}

#[no_mangle]
pub fn result_nop_try_macro(x: Result<i32, u32>) -> Result<i32, u32> {
// CHECK: start:
// CHECK-NEXT: ret i64 %0
    Ok(r#try!(x))
}

#[no_mangle]
pub fn result_nop_try_expr(x: Result<i32, u32>) -> Result<i32, u32> {
// CHECK: start:
// CHECK-NEXT: ret i64 %0
    Ok(x?)
}

#[no_mangle]
pub fn result_match_with_inlined_call(x: Result<i32, u32>) -> Result<i32, u32> {
// CHECK: start:
// CHECK-NEXT: ret i64 %0
    let y = match into_result(x) {
        Err(e) => return from_error(From::from(e)),
        Ok(v) => v,
    };
    Ok(y)
}

#[inline]
fn into_result<T, E>(r: Result<T, E>) -> Result<T, E> {
    r
}

#[inline]
fn from_error<T, E>(e: E) -> Result<T, E> {
    Err(e)
}
