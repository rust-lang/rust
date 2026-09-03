//@ compile-flags: -Copt-level=3 -g

#![crate_type = "lib"]

#[inline(always)]
fn foo() {
    bar();
}

#[inline(never)]
#[no_mangle]
fn bar() {
    panic!();
}

#[no_mangle]
pub fn example() {
    foo();
}

// CHECK-LABEL: @example
// CHECK:   tail call void @bar(){{( #[0-9]+)?}}, !dbg [[DBG_ID:![0-9]+]]
// CHECK-DAG: [[DBG_ID]] = !DILocation(line: 7, {{.*}}inlinedAt: [[INLINE_ID:![0-9]+]])
// CHECK-DAG: [[INLINE_ID]] = !DILocation(line: 18,
