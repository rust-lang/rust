// Tests that a disjunction passed to `unreachable_unchecked` still rules out
// both division operands, so neither the division-by-zero check nor the
// overflow check survives.
// See <https://github.com/rust-lang/rust/issues/115026>.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @disjunction_div(
// CHECK-NOT: panic
// CHECK-NOT: br {{.*}}
// CHECK: sdiv i64
// CHECK: ret i64
#[no_mangle]
pub fn disjunction_div(num: i64, x: i64) -> i64 {
    unsafe {
        if x == -1 || x == 0 {
            std::hint::unreachable_unchecked()
        }
    }
    num / x
}
