// Verify that we can pretty print invalid constant introduced
// by constant propagation. Regression test for issue #93688.
//
// compile-flags: -Copt-level=0 -Zinline-mir

#[inline(always)]
pub fn f(x: Option<Option<()>>) -> Option<()> {
    match x {
        None => None,
        Some(y) => y,
    }
}

// EMIT_MIR invalid_constant.main.ConstProp.diff
fn main() {
    f(None);
}
