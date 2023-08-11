// compile-flags: -O
// min-system-llvm-version: 17
// only-x86_64-unknown-linux-gnu

// We want to check that this function does not mis-optimize to loop jumping.

#![crate_type = "lib"]

#[repr(C)]
pub enum Expr {
    Sum,
    // must have more than usize data
    Sub(usize, u8),
}

#[no_mangle]
pub extern "C" fn issue_114312(expr: Expr) {
    // CHECK-LABEL: @issue_114312(
    // CHECK-SAME: readonly
    // CHECK-SAME: byval
    // CHECK: bb1:
    // CHECK-NEXT: br label %bb1
    match expr {
        Expr::Sum => {}
        Expr::Sub(_, _) => issue_114312(Expr::Sum),
    }
}
