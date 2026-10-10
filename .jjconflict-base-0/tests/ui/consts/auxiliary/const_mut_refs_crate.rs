// This is a support file for ../const-mut-refs-crate.rs

// This is to test that static inners from an external
// crate like this one, still preserves the alloc.
// That is, the address from the standpoint of rustc+llvm
// is the same.
// The need for this test originated from the GH issue
// https://github.com/rust-lang/rust/issues/57349

// See also ../const-mut-refs-crate.rs for more details
// about this test.

// if we used immutable references here, then promotion would
// turn the `&42` into a promoted, which gets duplicated arbitrarily.
pub static mut FOO: &'static mut i32 = &mut 42;
pub static mut BAR: &'static mut i32 = unsafe { FOO };

pub mod inner {
    pub static INNER_MOD_FOO: &'static i32 = &43;
    pub static INNER_MOD_BAR: &'static i32 = INNER_MOD_FOO;
}
