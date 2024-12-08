//@ compile-flags: --crate-type=lib -Copt-level=0 -Zmir-opt-level=0 -C debuginfo=2

// Test that we don't generate a memory allocation for the constant
// and read the fields from that, but instead just create the value pair directly.
pub fn foo() -> (i32, i32) {
    // CHECK: ret { i32, i32 } { i32 1, i32 2 }
    const { (1, 2) }
}
