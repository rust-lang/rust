//@ compile-flags: -g -Copt-level=3

// Check that simple constant values are preserved in debuginfo across both MIR opts and LLVM opts

#![crate_type = "lib"]

#[no_mangle]
pub fn check_it() {
    let a = 1;
    let b = 42;

    foo(a + b);
}

#[inline(never)]
fn foo(x: i32) {
    std::process::exit(x);
}

// CHECK-LABEL: @check_it
// CHECK: dbg{{.}}value({{(metadata )?}}i32 1, {{(metadata )?}}![[a_metadata:[0-9]+]], {{(metadata )?}}!DIExpression()
// CHECK: dbg{{.}}value({{(metadata )?}}i32 42, {{(metadata )?}}![[b_metadata:[0-9]+]], {{(metadata )?}}!DIExpression()

// CHECK: ![[a_metadata]] = !DILocalVariable(name: "a"
// CHECK-SAME: line: 9

// CHECK: ![[b_metadata]] = !DILocalVariable(name: "b"
// CHECK-SAME: line: 10
