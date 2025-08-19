//@ compile-flags: -Copt-level=3
//@ only-x86_64-unknown-linux-gnu

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
    // CHECK-SAME: byval
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    match expr {
        Expr::Sum => {}
        Expr::Sub(_, _) => issue_114312(Expr::Sum),
    }
}
