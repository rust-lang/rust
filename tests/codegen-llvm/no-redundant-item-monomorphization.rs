// Test to make sure that inner functions within a polymorphic outer function
// don't get re-codegened when the outer function is monomorphized. The test
// code monomorphizes the outer functions several times, but the magic constants
// used in the inner functions should each appear only once in the generated IR.

// issue: rust-lang/rust#7349
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0

// CHECK-COUNT-1: ret i32 8675309
// CHECK-COUNT-1: ret i32 11235813

fn outer<T>() {
    #[allow(dead_code)]
    fn inner() -> u32 {
        8675309
    }
    inner();
}

extern "C" fn outer_foreign<T>() {
    #[allow(dead_code)]
    fn inner() -> u32 {
        11235813
    }
    inner();
}

fn main() {
    outer::<isize>();
    outer::<usize>();
    outer_foreign::<isize>();
    outer_foreign::<usize>();
}
