// This test checks that debug information includes function argument indexes even if the function
// gets inlined by MIR inlining. Without function argument indexes, `info args` in gdb won't show
// arguments and their values for the current function.

//@ compile-flags: -Zinline-mir=yes -Cdebuginfo=2
//@ edition: 2021

#![crate_type = "lib"]

#[inline(never)]
pub fn outer_function(x: usize, y: usize) -> usize {
    inner_function(x, y) + 1
}

#[inline]
fn inner_function(aaaa: usize, bbbb: usize) -> usize {
    // CHECK: !DILocalVariable(name: "aaaa", arg: 1
    // CHECK-SAME: line: 16
    // CHECK-NOT: !DILexicalBlock(
    // CHECK: !DILocalVariable(name: "bbbb", arg: 2
    // CHECK-SAME: line: 16
    aaaa + bbbb
}
