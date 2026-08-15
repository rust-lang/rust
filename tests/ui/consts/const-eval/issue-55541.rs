//@ check-pass

// Test that we can handle newtypes wrapping extern types

#![feature(extern_types)]

use std::marker::PhantomData;

extern "C" {
  pub type ExternType;
}
unsafe impl Sync for ExternType {}
static MAGIC_FFI_STATIC: u8 = 42;

#[repr(transparent)]
pub struct Wrapper(ExternType);
pub static MAGIC_FFI_REF: &'static Wrapper = unsafe {
  std::mem::transmute(&MAGIC_FFI_STATIC)
};

#[repr(transparent)]
pub struct Wrapper2(PhantomData<Vec<i32>>, ExternType);
pub static MAGIC_FFI_REF2: &'static Wrapper2 = unsafe {
  std::mem::transmute(&MAGIC_FFI_STATIC)
};

fn main() {}
