pub mod discriminator;
pub mod llvm_siphash;

pub use discriminator::{
    FnPtrDiscriminatorSource, FnPtrTypeDiscriminatorInput, is_ptrauth_signed_abi,
    ptrauth_compute_fn_ptr_type_discriminator_for, ptrauth_signing_discriminator_for,
};
