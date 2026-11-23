// Checks that valid debuginfo is generated even when an inlined callee's parameter variable is
// within a `DISubprogram` in one place and within a `DILexicalBlockFile` in another place.

//@ compile-flags: -Zinline-mir=yes -Copt-level=0 -Cdebuginfo=2 -Ccodegen-units=1 -Zverify-llvm-ir
//@ edition: 2021

#![crate_type = "lib"]

#[macro_use]
#[path = "auxiliary/tricky_macros.rs"]
mod tricky_macros;

define_functions!();

#[inline(never)]
pub fn from_source(b: u32) -> u32 {
    // CHECK: !DILocalVariable(name: "x", arg: 1
    // CHECK-NOT: !DILocalVariable(name: "x", arg: 1
    callee(b)
}
