// compile-flags: -O -g

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
// CHECK: [[DBG_ID]] = !DILocation(line: 7,
// CHECK-SAME:                     inlinedAt: [[INLINE_ID:![0-9]+]])
// CHECK: [[INLINE_ID]] = !DILocation(line: 18,
