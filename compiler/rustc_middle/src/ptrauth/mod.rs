pub mod c_discriminator;
pub mod discriminator;
pub mod llvm_siphash;

pub use c_discriminator::{PtrauthCDiscriminatorError, ptrauth_c_discriminator_slots};
pub use discriminator::{
    FnPtrDiscriminatorSource, FnPtrTypeDiscriminatorInput, PtrauthCEncoding, is_ptrauth_signed_abi,
    ptrauth_c_encoding, ptrauth_compute_fn_ptr_type_discriminator_for,
    ptrauth_signing_discriminator_for,
};
