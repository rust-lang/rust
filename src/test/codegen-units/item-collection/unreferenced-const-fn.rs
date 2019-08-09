// compile-flags:-Zprint-mono-items=lazy

#![deny(dead_code)]
#![crate_type = "rlib"]

//~ MONO_ITEM fn unreferenced_const_fn::foo[0] @@ unreferenced_const_fn-cgu.0[External]
pub const fn foo(x: u32) -> u32 {
    x + 0xf00
}
