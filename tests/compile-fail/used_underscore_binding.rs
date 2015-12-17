#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy)]

/// Test that we lint if we use a binding with a single leading underscore
fn prefix_underscore(_x: u32) -> u32 {
    _x + 1 //~ ERROR used binding which is prefixed with an underscore
}

/// Test that we lint even if the use is within a macro expansion
fn in_macro(_x: u32) {
    println!("{}", _x); //~ ERROR used binding which is prefixed with an underscore
}

/// Test that we do not lint if the underscore is not a prefix
fn non_prefix_underscore(some_foo: u32) -> u32 {
    some_foo + 1
}

/// Test that we do not lint if we do not use the binding
fn unused_underscore(_foo: u32) -> u32 {
    1
}

// Non-variable bindings with preceding underscore
fn _fn_test() {}
struct _StructTest;
enum _EnumTest {
    _FieldA,
    _FieldB(_StructTest)
}

/// Test that we do not lint for non-variable bindings
fn non_variables() {
    _fn_test();
    let _s = _StructTest;
    let _e = match _EnumTest::_FieldB(_StructTest) {
        _EnumTest::_FieldA => 0,
        _EnumTest::_FieldB(_st) => 1,
    };
    let f = _fn_test;
    f();
}

fn main() {
    let foo = 0u32;
    // tests of unused_underscore lint
    let _ = prefix_underscore(foo);
    in_macro(foo);
    // possible false positives
    let _ = non_prefix_underscore(foo);
    let _ = unused_underscore(foo);
    non_variables();
}

