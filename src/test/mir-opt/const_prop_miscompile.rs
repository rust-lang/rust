#![feature(raw_ref_op)]

// EMIT_MIR rustc.foo.ConstProp.diff
fn foo() {
    let mut u = (1,);
    *&mut u.0 = 5;
    let y = { u.0 } == 5;
}

// EMIT_MIR rustc.bar.ConstProp.diff
fn bar() {
    let mut v = (1,);
    unsafe {
        *&raw mut v.0 = 5;
    }
    let y = { v.0 } == 5;
}

fn main() {
    foo();
    bar();
}
