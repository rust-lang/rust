pub mod discriminator;
pub mod llvm_siphash;

pub use discriminator::{
    FnPtrDiscriminatorSource, FnPtrTypeDiscriminatorInput, ptrauth_clone_discriminated_schema_for,
    ptrauth_compute_fn_ptr_type_discriminator_for,
};
