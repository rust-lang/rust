// This test checks that debug information includes function argument indexes even if the function
// gets inlined by MIR inlining. Without function argument indexes, `info args` in gdb won't show
// arguments and their values for the current function. It also checks that a function inlined
// multiple times only gets a single debuginfo function entry.

//@ compile-flags: -Zinline-mir=yes -Cdebuginfo=2 -Ccodegen-units=1
//@ edition: 2021

#![crate_type = "lib"]

#[inline(never)]
pub fn outer_function(x: usize, y: usize) -> usize {
    inner_function(x, y) + 1
}

#[inline(never)]
pub fn outer_function2(x: usize, y: usize) -> usize {
    inner_function(x, y) + 2
}

#[inline]
fn inner_function(aaaa: usize, bbbb: usize) -> usize {
    // Check the following:
    // - Exactly one `aaaa` variable entry.
    // - Exactly one `bbbb` variable entry.
    // - Exactly one `inner_function` function entry. (It comes between the `aaaa` and `bbbb`
    //   entries, curiously enough.)
    // - No lexical block entries between `aaaa` and `bbbb` entries.
    //
    // CHECK: !DILocalVariable(name: "aaaa", arg: 1
    // CHECK-SAME: line: 22
    // CHECK-NOT: !DILexicalBlock(
    // CHECK: !DISubprogram(name: "inner_function"
    // CHECK-NOT: !DILexicalBlock(
    // CHECK: !DILocalVariable(name: "bbbb", arg: 2
    // CHECK-SAME: line: 22
    // CHECK-NOT: !DISubprogram(name: "inner_function"
    // CHECK-NOT: !DILocalVariable(name: "aaaa", arg: 1
    // CHECK-NOT: !DILocalVariable(name: "bbbb", arg: 2
    aaaa + bbbb
}
