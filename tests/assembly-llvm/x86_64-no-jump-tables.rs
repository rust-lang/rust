// Test that jump tables are (not) emitted when the `-Zno-jump-tables`
// flag is (not) set.

//@ revisions: unset set
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ [set] compile-flags: -Zno-jump-tables
//@ only-x86_64
//@ ignore-sgx

#![crate_type = "lib"]

extern "C" {
    fn bar1();
    fn bar2();
    fn bar3();
    fn bar4();
    fn bar5();
    fn bar6();
}

// CHECK-LABEL: foo:
#[no_mangle]
pub unsafe fn foo(x: i32) {
    // unset: LJTI0_0
    // set-NOT: LJTI0_0
    match x {
        1 => bar1(),
        2 => bar2(),
        3 => bar3(),
        4 => bar4(),
        5 => bar5(),
        _ => bar6(),
    }
}
