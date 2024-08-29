//@ test-mir-pass: GVN
// Check that we do not propagate past an indirect mutation.

// EMIT_MIR indirect_mutation.foo.GVN.diff
fn foo() {
    // CHECK-LABEL: fn foo(
    // CHECK: debug u => _1;
    // CHECK: debug y => _3;
    // CHECK: _1 = const (1_i32,);
    // CHECK: _2 = &mut (_1.0: i32);
    // CHECK: (*_2) = const 5_i32;
    // CHECK: _4 = copy (_1.0: i32);
    // CHECK: _3 = Eq(move _4, const 5_i32);

    let mut u = (1,);
    *&mut u.0 = 5;
    let y = { u.0 } == 5;
}

// EMIT_MIR indirect_mutation.bar.GVN.diff
fn bar() {
    // CHECK-LABEL: fn bar(
    // CHECK: debug v => _1;
    // CHECK: debug y => _4;
    // CHECK: _3 = &raw mut (_1.0: i32);
    // CHECK: (*_3) = const 5_i32;
    // CHECK: _5 = copy (_1.0: i32);
    // CHECK: _4 = Eq(move _5, const 5_i32);

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
