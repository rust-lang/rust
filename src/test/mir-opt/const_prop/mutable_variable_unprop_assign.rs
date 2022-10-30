// unit-test
// compile-flags: -O

// EMIT_MIR mutable_variable_unprop_assign.main.ConstProp.diff
fn main() {
    let a = foo();
    let mut x: (i32, i32) = (1, 2);
    x.1 = a;
    let y = x.1;
    let z = x.0; // this could theoretically be allowed, but we can't handle it right now
}

#[inline(never)]
fn foo() -> i32 {
    unimplemented!()
}
