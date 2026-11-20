// The discriminator of `#[ptrauth_c_discriminator]` is 16 bits wide.

#![feature(ptrauth_c_discriminator)]
#![crate_type = "lib"]

pub struct NativeMethod {
    #[ptrauth_c_discriminator(65536)] //~ ERROR malformed `ptrauth_c_discriminator` attribute input
    pub fn_ptr: *const u8,
}
