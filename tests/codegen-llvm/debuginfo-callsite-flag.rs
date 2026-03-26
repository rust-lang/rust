// Check that DIFlagAllCallsDescribed is set on subprogram definitions.

//@ ignore-msvc (CodeView does not use DIFlagAllCallsDescribed)
//@ compile-flags: -C debuginfo=2 -C opt-level=1 -C no-prepopulate-passes

// CHECK: {{.*}}DISubprogram{{.*}}name: "foo"{{.*}}DIFlagAllCallsDescribed{{.*}}

#[no_mangle]
#[inline(never)]
pub fn foo(x: i32) -> i32 {
    bar(x + 1)
}

#[no_mangle]
#[inline(never)]
pub fn bar(x: i32) -> i32 {
    x * 2
}

fn main() {}
