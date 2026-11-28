#![crate_type = "lib"]

pub struct NativeMethod {
    #[rustc_ptrauth_c_discriminator(0)] //~ ERROR the `rustc_ptrauth_c_discriminator` attribute is an experimental feature [E0658]
    pub fn_ptr: *const u8,
}
