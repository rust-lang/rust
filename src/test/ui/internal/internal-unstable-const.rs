// Don't allow unstable features in stable functions without `allow_internal_unstable`.

#![stable(feature = "rust1", since = "1.0.0")]

#![feature(staged_api)]
#![feature(const_transmute, const_fn)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
pub const fn foo() -> i32 {
    unsafe { std::mem::transmute(4u32) } //~ ERROR can only call `transmute` from const items
}

fn main() {}
