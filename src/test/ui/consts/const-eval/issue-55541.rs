// compile-pass

// Test that we can handle newtypes wrapping extern types

#![feature(extern_types, const_transmute)]

extern "C" {
  pub type ExternType;
}
unsafe impl Sync for ExternType {}

#[repr(transparent)]
pub struct Wrapper(ExternType);

static MAGIC_FFI_STATIC: u8 = 42;

pub static MAGIC_FFI_REF: &'static Wrapper = unsafe {
  std::mem::transmute(&MAGIC_FFI_STATIC)
};

fn main() {}
