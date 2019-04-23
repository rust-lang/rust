// compile-flags:-Zprint-mono-items=lazy

// N.B., we do not expect *any* monomorphization to be generated here.

#![deny(dead_code)]
#![crate_type = "rlib"]

#[inline]
pub fn foo() -> bool {
    [1, 2] == [3, 4]
}
