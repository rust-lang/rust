// The discriminator of `#[rustc_ptrauth_c_discriminator]` is 16 bits wide.

#![feature(ptrauth_c_discriminator)]
#![crate_type = "lib"]

pub struct NativeMethod {
    #[rustc_ptrauth_c_discriminator(65536)] //~ ERROR malformed `rustc_ptrauth_c_discriminator` attribute input
    pub fn_ptr: *const u8,
}
