//! Verify that we manage to propagate the value of aggregate `a` even without directly mentioning
//! the contained scalars.
//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes

const Foo: (u32, u32) = (5, 3);

fn foo() -> u32 {
    // CHECK-LABEL: fn foo(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: debug c => [[c:_.*]];

    // CHECK:bb0: {
    // CHECK:    [[a]] = const Foo;
    // CHECK:    [[b]] = const (5_u32, 3_u32);
    // CHECK:    [[c]] = const 3_u32;
    // CHECK:    {{_.*}} = const 3_u32;
    // CHECK:    {{_.*}} = const true;
    // CHECK:    switchInt(const true) -> [0: bb2, otherwise: bb1];

    // CHECK:bb1: {
    // CHECK:    _0 = const 5_u32;
    // CHECK:    goto -> bb3;

    // CHECK:bb2: {
    // CHECK:    _0 = const 13_u32;
    // CHECK:    goto -> bb3;

    let a = Foo;
    // This copies the struct in `a`. We want to ensure that we do track the contents of `a`
    // because we need to read `b` later.
    let b = a;
    let c = b.1;
    if c >= 2 { b.0 } else { 13 }
}

fn main() {
    // CHECK-LABEL: fn main(
    foo();
}

// EMIT_MIR aggregate_copy.foo.DataflowConstProp.diff
