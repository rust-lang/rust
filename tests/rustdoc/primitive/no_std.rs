#![no_std]
#![deny(warnings)]
#![deny(rustdoc::broken_intra_doc_links)]

// @has no_std/fn.foo.html '//a/[@href="{{channel}}/core/primitive.u8.html"]' 'u8'
// @has no_std/fn.foo.html '//a/[@href="{{channel}}/core/primitive.u8.html"]' 'primitive link'
/// Link to [primitive link][u8]
pub fn foo() -> u8 {}

// Test that all primitives can be linked to.
/// [isize] [i8] [i16] [i32] [i64] [i128]
/// [usize] [u8] [u16] [u32] [u64] [u128]
/// [f32] [f64]
/// [char] [bool] [str] [slice] [array] [tuple] [unit]
/// [pointer] [reference] [fn] [never]
pub fn bar() {}
